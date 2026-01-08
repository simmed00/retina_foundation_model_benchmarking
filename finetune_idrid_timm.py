"""
IDRID 数据集微调程序
基于 timm 模型（ImageNet 预训练权重）进行微调
支持五分类和二分类（转诊/非转诊）任务
记录每个 epoch 的 val 和 test 指标：acc, auc, auprc, f1
"""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import tqdm
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import math
import timm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class IDRIDDataset(Dataset):
    """IDRID 数据集类，适配 CSV 格式（filename, label）"""
    def __init__(self, csv_file, data_path, is_train=True, mode='train'):
        """
        Args:
            csv_file: CSV 文件路径，包含 filename 和 label 列
            data_path: 图片根目录路径
            is_train: 是否为训练集（决定是否使用数据增强）
            mode: 数据集模式，'train' 或 'val' 在 Training Set 下查找，'test' 在 Testing Set 下查找
        """
        self.df = pd.read_csv(csv_file)
        self.data_path = Path(data_path)
        self.is_train = is_train
        self.mode = mode  # 'train', 'val', 'test'
        
        # 检查列名
        if 'filename' in self.df.columns:
            self.filenames = self.df['filename'].tolist()
        elif 'Image name' in self.df.columns:
            self.filenames = self.df['Image name'].tolist()
        else:
            raise ValueError(f"CSV 文件必须包含 'filename' 或 'Image name' 列，当前列: {self.df.columns.tolist()}")
        
        if 'label' in self.df.columns:
            self.labels = self.df['label'].tolist()
        elif 'Retinopathy grade' in self.df.columns:
            self.labels = self.df['Retinopathy grade'].tolist()
        else:
            raise ValueError(f"CSV 文件必须包含 'label' 或 'Retinopathy grade' 列，当前列: {self.df.columns.tolist()}")
        
        # 数据增强
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224),scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        label = int(self.labels[index])
        
        # 移除可能的扩展名
        filename_base = str(filename).replace('.jpg', '').replace('.JPG', '').replace('.png', '').replace('.PNG', '')
        
        # 根据 mode 决定在哪个文件夹下查找
        # train 和 val 在 Training Set 下查找，test 在 Testing Set 下查找
        if self.mode in ['train', 'val']:
            # 训练集和验证集在 Training Set 下查找
            possible_paths = [
                self.data_path / "a. Training Set" / f"{filename_base}.jpg",
                self.data_path / "a. Training Set" / f"{filename_base}.JPG",
                self.data_path / "1. Original Images" / "a. Training Set" / f"{filename_base}.jpg",
                self.data_path / "1. Original Images" / "a. Training Set" / f"{filename_base}.JPG",
                self.data_path / "Training Set" / f"{filename_base}.jpg",
                self.data_path / "Training Set" / f"{filename_base}.JPG",
            ]
        else:  # test
            # 测试集在 Testing Set 下查找
            possible_paths = [
                self.data_path / "b. Testing Set" / f"{filename_base}.jpg",
                self.data_path / "b. Testing Set" / f"{filename_base}.JPG",
                self.data_path / "1. Original Images" / "b. Testing Set" / f"{filename_base}.jpg",
                self.data_path / "1. Original Images" / "b. Testing Set" / f"{filename_base}.JPG",
                self.data_path / "Testing Set" / f"{filename_base}.jpg",
                self.data_path / "Testing Set" / f"{filename_base}.JPG",
            ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"找不到图片文件: {filename} (mode={self.mode}), 尝试过的路径: {possible_paths[:3]}")
        
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except Exception as e:
            print(f"无法打开图片 {img_path}: {e}")
            raise
        
        img = self.transform(img)
        
        return img, label, filename


class Model_Finetuning(torch.nn.Module):
    """微调模型：使用 timm 模型（ImageNet 预训练权重） + 分类头"""
    def __init__(self, model_name, class_num, pretrained=True):
        super().__init__()
        
        # 使用 timm 创建模型，加载 ImageNet 预训练权重
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # 移除原始分类头
                global_pool='avg'  # 使用全局平均池化
            )
            print(f"成功加载 timm 模型: {model_name}, pretrained={pretrained}")
        except Exception as e:
            raise ValueError(f"无法创建 timm 模型 '{model_name}': {e}\n"
                           f"请检查模型名称是否正确，可用模型列表请参考: timm.list_models()")
        
        # 获取特征维度
        # 通过前向传播获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            # 处理不同的输出格式
            if dummy_output.dim() == 1:
                feature_dim = dummy_output.shape[0]
            elif dummy_output.dim() == 2:
                feature_dim = dummy_output.shape[1]
            else:
                # 如果是多维，展平后取最后一个维度
                feature_dim = dummy_output.view(dummy_output.size(0), -1).shape[1]
        
        print(f"模型特征维度: {feature_dim}")
        
        # 分类头
        self.classifier = torch.nn.Linear(feature_dim, class_num, bias=True)
        
        # 初始化分类头
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        # 提取特征
        x_features = self.backbone(x)
        # 分类
        out = self.classifier(x_features)
        return out


def convert_to_binary_labels(labels):
    """
    将五分类标签转换为二分类标签（转诊/非转诊）
    dr0, dr1 -> 0 (非转诊)
    dr2, dr3, dr4 -> 1 (需转诊)
    """
    binary_labels = []
    for label in labels:
        if label in [0, 1]:
            binary_labels.append(0)
        elif label in [2, 3, 4]:
            binary_labels.append(1)
        else:
            raise ValueError(f"未知的标签值: {label}")
    return np.array(binary_labels)


def calculate_metrics_multi_class(y_true, y_pred, y_proba, num_classes=5):
    """
    计算多分类指标
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（每个类别的概率）
    Returns:
        dict: 包含 acc, auc, auprc, f1 的字典
    """
    # Accuracy
    acc = metrics.accuracy_score(y_true, y_pred)
    
    # AUC (macro average for multi-class)
    try:
        if num_classes == 2:
            # 二分类使用 roc_auc_score
            auc = metrics.roc_auc_score(y_true, y_proba[:, 1])
        else:
            # 多分类使用 one-vs-rest 策略
            auc = metrics.roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"计算 AUC 时出错: {e}")
        auc = 0.0
    
    # AUPRC (Average Precision)
    try:
        if num_classes == 2:
            auprc = metrics.average_precision_score(y_true, y_proba[:, 1])
        else:
            # 多分类使用 one-vs-rest 策略，然后取平均
            from sklearn.preprocessing import label_binarize
            y_true_binarized = label_binarize(y_true, classes=range(num_classes))
            if y_true_binarized.shape[1] == 1:
                # 如果只有一个类别，需要添加另一个类别
                y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
            auprc = metrics.average_precision_score(
                y_true_binarized,
                y_proba,
                average='macro'
            )
    except Exception as e:
        print(f"计算 AUPRC 时出错: {e}")
        auprc = 0.0
    
    # F1 score
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    
    return {
        'acc': acc,
        'auc': auc,
        'auprc': auprc,
        'f1': f1
    }


def evaluate(dataloader, model, epoch, args, mode, save_dir):
    """
    评估函数，计算五分类和二分类两种任务的指标
    """
    print(f'\n====== Start {mode} Evaluation ======')
    model.eval()
    
    all_labels = []
    all_predictions_5class = []
    all_probabilities_5class = []
    
    tbar = tqdm.tqdm(dataloader, desc=f'{mode}')
    
    with torch.no_grad():
        for img_data_list in tbar:
            Fundus_img = img_data_list[0].cuda()
            cls_label = img_data_list[1].long().cuda()
            
            pred = model.forward(Fundus_img)
            pred_proba = torch.softmax(pred, dim=1)
            pred_decision = pred_proba.argmax(dim=-1)
            
            all_labels.extend(cls_label.cpu().numpy())
            all_predictions_5class.extend(pred_decision.cpu().numpy())
            all_probabilities_5class.extend(pred_proba.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions_5class = np.array(all_predictions_5class)
    all_probabilities_5class = np.array(all_probabilities_5class)
    
    # 计算五分类指标
    metrics_5class = calculate_metrics_multi_class(
        all_labels, all_predictions_5class, all_probabilities_5class, num_classes=5
    )
    
    # 转换为二分类标签
    binary_labels = convert_to_binary_labels(all_labels)
    binary_predictions = convert_to_binary_labels(all_predictions_5class)
    
    # 计算二分类概率（将 dr0,dr1 合并为类别0，dr2,dr3,dr4 合并为类别1）
    binary_proba = np.zeros((len(binary_labels), 2))
    binary_proba[:, 0] = all_probabilities_5class[:, 0] + all_probabilities_5class[:, 1]  # dr0 + dr1
    binary_proba[:, 1] = (
        all_probabilities_5class[:, 2] + 
        all_probabilities_5class[:, 3] + 
        all_probabilities_5class[:, 4]
    )  # dr2 + dr3 + dr4
    
    # 计算二分类指标
    metrics_binary = calculate_metrics_multi_class(
        binary_labels, binary_predictions, binary_proba, num_classes=2
    )
    
    # 打印结果
    print(f"\n{mode} - 五分类指标:")
    print(f"  ACC: {metrics_5class['acc']:.6f}")
    print(f"  AUC: {metrics_5class['auc']:.6f}")
    print(f"  AUPRC: {metrics_5class['auprc']:.6f}")
    print(f"  F1: {metrics_5class['f1']:.6f}")
    
    print(f"\n{mode} - 二分类指标（转诊/非转诊）:")
    print(f"  ACC: {metrics_binary['acc']:.6f}")
    print(f"  AUC: {metrics_binary['auc']:.6f}")
    print(f"  AUPRC: {metrics_binary['auprc']:.6f}")
    print(f"  F1: {metrics_binary['f1']:.6f}")
    
    # 保存结果到文件
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(save_dir, f"{mode}_metrics.txt")
    
    with open(metrics_file, 'a+', encoding='utf-8') as f:
        f.write(f"\nEpoch {epoch} - {mode} Metrics:\n")
        f.write(f"五分类 - ACC: {metrics_5class['acc']:.6f}, AUC: {metrics_5class['auc']:.6f}, "
                f"AUPRC: {metrics_5class['auprc']:.6f}, F1: {metrics_5class['f1']:.6f}\n")
        f.write(f"二分类 - ACC: {metrics_binary['acc']:.6f}, AUC: {metrics_binary['auc']:.6f}, "
                f"AUPRC: {metrics_binary['auprc']:.6f}, F1: {metrics_binary['f1']:.6f}\n")
    
    # 保存 JSON 格式的详细结果
    json_file = os.path.join(save_dir, f"{mode}_metrics_epoch_{epoch}.json")
    results = {
        'epoch': epoch,
        'mode': mode,
        '5class': metrics_5class,
        'binary': metrics_binary
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    torch.cuda.empty_cache()
    
    return metrics_5class, metrics_binary


def train(train_loader, val_loader, test_loader, model, optimizer, criterion, args):
    """训练函数"""
    step = 0
    model = model.cuda()
    best_val_auc = 0.0
    scaler = GradScaler()

    # 学习率调度器（按 epoch 更新）
    def get_lr_scheduler(optimizer):
        warmup_epochs = max(0, int(args.warmup_epochs))
        total_epochs = int(args.num_epochs)

        if args.scheduler_type == "warmup_cosine":
            # 先线性 warmup，再 cosine decay 到 0
            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                if total_epochs == warmup_epochs:
                    return 1.0
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

        elif args.scheduler_type == "warmup_linear":
            # 先线性 warmup，再线性衰减到 0
            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                if total_epochs == warmup_epochs:
                    return 1.0
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                return max(0.0, 1.0 - progress)

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

        elif args.scheduler_type == "warmup_step":
            # 先 warmup，再在 50% 和 75% 训练进度处进行阶梯衰减
            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                factor = 1.0
                if epoch >= 0.5 * total_epochs:
                    factor *= 0.1
                if epoch >= 0.75 * total_epochs:
                    factor *= 0.1
                return factor

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

        else:
            return None

    scheduler = get_lr_scheduler(optimizer)
    
    # 创建保存目录
    save_dir = os.path.join(args.save_model_path, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建CSV文件记录每个epoch的指标
    csv_file = os.path.join(save_dir, "training_metrics.csv")
    
    # 写入配置信息和表头
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        import csv as csv_module
        writer = csv_module.writer(f)
        # 写入配置信息
        writer.writerow(['# Training Configuration'])
        writer.writerow(['Model', args.model_name])
        writer.writerow(['Batch Size', args.batch_size])
        writer.writerow(['Image Size', '224'])  # 固定为224
        writer.writerow(['Initial LR', args.lr])
        writer.writerow(['Scheduler', args.scheduler_type])
        writer.writerow(['Warmup Epochs', args.warmup_epochs])
        writer.writerow(['Total Epochs', args.num_epochs])
        writer.writerow(['Num Classes', args.num_classes])
        writer.writerow(['# Training Metrics'])
        # 写入表头
        writer.writerow([
            'Epoch', 'LR', 'Train_Loss',
            'Val_5class_ACC', 'Val_5class_AUC', 'Val_5class_AUPRC', 'Val_5class_F1',
            'Val_Binary_ACC', 'Val_Binary_AUC', 'Val_Binary_AUPRC', 'Val_Binary_F1',
            'Test_5class_ACC', 'Test_5class_AUC', 'Test_5class_AUPRC', 'Test_5class_F1',
            'Test_Binary_ACC', 'Test_Binary_AUC', 'Test_Binary_AUPRC', 'Test_Binary_F1'
        ])
    
    # 记录所有 epoch 的结果
    all_results = []
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练阶段
        model.train()
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
        tq.set_description(f'Epoch {epoch}/{args.num_epochs}, lr {args.lr:.6f}')
        loss_record = []
        train_loss = 0.0
        
        for i, img_data_list in enumerate(train_loader):
            Fundus_img = img_data_list[0].cuda(non_blocking=True)
            cls_label = img_data_list[1].long().cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                predict = model.forward(Fundus_img)
                loss_CE = criterion(predict, cls_label)
                loss = loss_CE

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss=f'{train_loss / (i + 1):.6f}')
            step += 1
            loss_record.append(loss.item())
        
        tq.close()
        torch.cuda.empty_cache()
        loss_train_mean = np.mean(loss_record)
        
        # 当前学习率（打印第一个 param group 的 lr）
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch {epoch} - Train Loss: {loss_train_mean:.6f} - LR: {current_lr:.6f}')
        
        # 验证阶段
        val_metrics_5class, val_metrics_binary = evaluate(
            val_loader, model, epoch, args, mode="val", save_dir=save_dir
        )
        
        # 测试阶段（每个 epoch 都评估）
        test_metrics_5class, test_metrics_binary = evaluate(
            test_loader, model, epoch, args, mode="test", save_dir=save_dir
        )
        
        # 记录结果
        epoch_result = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': loss_train_mean,
            'val_5class': val_metrics_5class,
            'val_binary': val_metrics_binary,
            'test_5class': test_metrics_5class,
            'test_binary': test_metrics_binary
        }
        all_results.append(epoch_result)
        
        # 追加到CSV文件
        with open(csv_file, 'a', encoding='utf-8-sig', newline='') as f:
            import csv as csv_module
            writer = csv_module.writer(f)
            writer.writerow([
                epoch, f'{current_lr:.8f}', f'{loss_train_mean:.6f}',
                f'{val_metrics_5class["acc"]:.6f}', f'{val_metrics_5class["auc"]:.6f}', 
                f'{val_metrics_5class["auprc"]:.6f}', f'{val_metrics_5class["f1"]:.6f}',
                f'{val_metrics_binary["acc"]:.6f}', f'{val_metrics_binary["auc"]:.6f}', 
                f'{val_metrics_binary["auprc"]:.6f}', f'{val_metrics_binary["f1"]:.6f}',
                f'{test_metrics_5class["acc"]:.6f}', f'{test_metrics_5class["auc"]:.6f}', 
                f'{test_metrics_5class["auprc"]:.6f}', f'{test_metrics_5class["f1"]:.6f}',
                f'{test_metrics_binary["acc"]:.6f}', f'{test_metrics_binary["auc"]:.6f}', 
                f'{test_metrics_binary["auprc"]:.6f}', f'{test_metrics_binary["f1"]:.6f}'
            ])
        
        # 保存最佳模型（基于 AUC）
        is_best = val_metrics_5class['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics_5class['auc']
            print(f'New best validation AUC: {best_val_auc:.6f}')
            
            checkpoint_dir = args.save_model_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 删除之前的模型
            import glob
            all_previous_models = glob.glob(os.path.join(checkpoint_dir, "*.pth.tar"))
            if len(all_previous_models):
                for pre_model in all_previous_models:
                    os.remove(pre_model)
            
            print('===> Saving best model...')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_auc': best_val_auc,
                'val_metrics_5class': val_metrics_5class,
                'val_metrics_binary': val_metrics_binary,
                'test_metrics_5class': test_metrics_5class,
                'test_metrics_binary': test_metrics_binary,
            }, os.path.join(checkpoint_dir, "checkpoint_best.pth.tar"))
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 保存所有 epoch 的结果摘要
        summary_file = os.path.join(save_dir, "training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f'\n训练完成！最佳验证集 AUC: {best_val_auc:.6f}')


def main(args):
    """主函数"""
    # 构建模型
    model = Model_Finetuning(
        model_name=args.model_name,
        class_num=args.num_classes,
        pretrained=args.pretrained
    )
    
    # 准备数据集
    data_path = args.data_path
    csv_path = args.csv_path
    
    train_csv = os.path.join(csv_path, "train.csv")
    val_csv = os.path.join(csv_path, "val.csv")
    test_csv = os.path.join(csv_path, "test.csv")
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"训练集 CSV 文件不存在: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"验证集 CSV 文件不存在: {val_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"测试集 CSV 文件不存在: {test_csv}")
    
    train_dataset = IDRIDDataset(csv_file=train_csv, data_path=data_path, is_train=True, mode='train')
    train_loader = DataLoader(
        train_dataset,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    val_dataset = IDRIDDataset(csv_file=val_csv, data_path=data_path, is_train=False, mode='val')
    val_loader = DataLoader(
        val_dataset,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    test_dataset = IDRIDDataset(csv_file=test_csv, data_path=data_path, is_train=False, mode='test')
    test_loader = DataLoader(
        test_dataset,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda()
    
    # 开始训练
    train(train_loader, val_loader, test_loader, model, optimizer, criterion, args)


def get_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='IDRID 数据集微调（基于 timm 模型）')
    
    parser.add_argument("--num_classes", type=int, default=5, help="分类类别数（IDRID 为 5 类）")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--save_model_path", type=str, default="./checkpoints/idrid_timm", help="模型保存路径")
    parser.add_argument("--model_name", type=str, default="efficientnet_b3", 
                        help="timm 模型名称，例如: convnext_base, efficientnet_b3, resnet50 等")
    parser.add_argument("--pretrained", action='store_true', default=True,
                        help="是否使用 ImageNet 预训练权重（默认 True）")
    parser.add_argument("--no_pretrained", dest='pretrained', action='store_false',
                        help="不使用 ImageNet 预训练权重")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--scheduler_type", type=str, default="warmup_cosine",
                        choices=["warmup_cosine", "warmup_linear", "warmup_step"],
                        help="学习率调度策略，默认 warmup_cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="warmup 轮数（按 epoch 计），0 表示不使用 warmup")
    parser.add_argument("--data_path", type=str, default="I:/Dataset/IDRID/Disease Grading/Original Images", help="图片数据路径")
    parser.add_argument("--csv_path", type=str, default="./idrid-split", help="CSV 文件路径")
    
    return parser


if __name__ == "__main__":
    """
    使用示例:
    
    # 基本使用（使用默认参数，ConvNeXt-Base）
    python finetune_idrid_timm.py
    
    # 使用 EfficientNet-B3
    python finetune_idrid_timm.py \
        --model_name efficientnet_b3 \
        --data_path "I:/Dataset/IDRID/Disease Grading/Original Images" \
        --csv_path "./idrid-split" \
        --batch_size 32 \
        --lr 5e-4 \
        --num_epochs 100 \
        --save_model_path ./checkpoints/idrid_efficientnet
    
    # 使用 ResNet-50
    python finetune_idrid_timm.py \
        --model_name resnet50 \
        --batch_size 64 \
        --lr 1e-3
    """
    torch.set_num_threads(4)
    parser = get_parser()
    args = parser.parse_args()
    args.seed = 1234
    
    # 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("="*60)
    print("IDRID 数据集微调（基于 timm 模型）")
    print("="*60)
    print(f"模型名称: {args.model_name}")
    print(f"使用预训练权重: {args.pretrained}")
    print(f"数据路径: {args.data_path}")
    print(f"CSV 路径: {args.csv_path}")
    print(f"类别数: {args.num_classes}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率调度: {args.scheduler_type}, warmup_epochs: {args.warmup_epochs}")
    print("="*60)
    
    main(args)

