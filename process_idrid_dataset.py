"""
IDRID 数据集处理程序
从训练集中划分验证集（20%），保持各类别比例一致，并生成 train/val/test 三个 CSV 文件
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def process_idrid_dataset(
    dataset_root: str,
    train_labels_file: str = None,
    test_labels_file: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42
):
    """
    处理 IDRID 数据集，划分训练集和验证集，生成 CSV 文件
    
    Args:
        dataset_root: 数据集根目录路径 (例如: I:\Dataset\IDRID\Disease Grading)
        train_labels_file: 训练集标签文件路径（如果为 None，则自动查找）
        test_labels_file: 测试集标签文件路径（如果为 None，则自动查找）
        val_ratio: 验证集占原始训练集的比例（默认 0.2，即 20%）
        random_state: 随机种子，确保结果可复现
    """
    dataset_root = Path(dataset_root)
    
    # 自动查找标签文件
    # 尝试多种可能的目录名
    possible_gt_dirs = [
        dataset_root / "2. Groundtruths",
        dataset_root / "Groundtruths",
        dataset_root / "groundtruths",
        dataset_root / "Ground Truth",
        dataset_root / "ground truth",
    ]
    
    groundtruth_dir = None
    for gt_dir in possible_gt_dirs:
        if gt_dir.exists():
            groundtruth_dir = gt_dir
            break
    
    if groundtruth_dir and groundtruth_dir.exists():
        if train_labels_file is None:
            # 查找训练集标签文件（尝试多种命名模式）
            train_patterns = [
                "*Training*Labels*.csv",
                "*training*labels*.csv",
                "*Train*Labels*.csv",
                "*train*labels*.csv",
                "a.*.csv",
            ]
            train_files = []
            for pattern in train_patterns:
                train_files = list(groundtruth_dir.glob(pattern))
                if train_files:
                    break
            if train_files:
                train_labels_file = str(train_files[0])
            else:
                raise FileNotFoundError(f"未找到训练集标签文件，请检查 {groundtruth_dir}")
        
        if test_labels_file is None:
            # 查找测试集标签文件（尝试多种命名模式）
            test_patterns = [
                "*Testing*Labels*.csv",
                "*testing*labels*.csv",
                "*Test*Labels*.csv",
                "*test*labels*.csv",
                "b.*.csv",
            ]
            test_files = []
            for pattern in test_patterns:
                test_files = list(groundtruth_dir.glob(pattern))
                if test_files:
                    break
            if test_files:
                test_labels_file = str(test_files[0])
            else:
                raise FileNotFoundError(f"未找到测试集标签文件，请检查 {groundtruth_dir}")
    else:
        if train_labels_file is None or test_labels_file is None:
            raise FileNotFoundError(
                f"未找到 Groundtruths 目录。已尝试: {[str(d) for d in possible_gt_dirs]}\n"
                f"请手动指定 train_labels_file 和 test_labels_file 参数"
            )
    
    print(f"训练集标签文件: {train_labels_file}")
    print(f"测试集标签文件: {test_labels_file}")
    
    # 读取标签文件
    print("\n正在读取标签文件...")
    df_train = pd.read_csv(train_labels_file)
    df_test = pd.read_csv(test_labels_file)
    
    print(f"训练集样本数: {len(df_train)}")
    print(f"测试集样本数: {len(df_test)}")
    
    # 检查列名
    if 'Image name' not in df_train.columns:
        # 尝试其他可能的列名
        possible_cols = [col for col in df_train.columns if 'image' in col.lower() or 'name' in col.lower()]
        if possible_cols:
            df_train = df_train.rename(columns={possible_cols[0]: 'Image name'})
        else:
            raise ValueError(f"未找到图片名称列，可用列: {df_train.columns.tolist()}")
    
    if 'Retinopathy grade' not in df_train.columns:
        # 尝试其他可能的列名
        possible_cols = [col for col in df_train.columns if 'grade' in col.lower() or 'label' in col.lower()]
        if possible_cols:
            df_train = df_train.rename(columns={possible_cols[0]: 'Retinopathy grade'})
            df_test = df_test.rename(columns={possible_cols[0]: 'Retinopathy grade'})
        else:
            raise ValueError(f"未找到标签列，可用列: {df_train.columns.tolist()}")
    
    # 显示类别分布
    print("\n原始训练集类别分布:")
    train_class_dist = df_train['Retinopathy grade'].value_counts().sort_index()
    for grade, count in train_class_dist.items():
        print(f"  类别 {grade}: {count} 个样本 ({count/len(df_train)*100:.2f}%)")
    
    print("\n测试集类别分布:")
    test_class_dist = df_test['Retinopathy grade'].value_counts().sort_index()
    for grade, count in test_class_dist.items():
        print(f"  类别 {grade}: {count} 个样本 ({count/len(df_test)*100:.2f}%)")
    
    # 按类别分层划分训练集和验证集
    print(f"\n正在从训练集中划分验证集（{val_ratio*100:.1f}%）...")
    
    train_list = []
    val_list = []
    
    # 获取所有类别
    unique_classes = sorted(df_train['Retinopathy grade'].unique())
    
    for class_label in unique_classes:
        class_data = df_train[df_train['Retinopathy grade'] == class_label]
        class_images = class_data['Image name'].tolist()
        
        if len(class_images) == 0:
            continue
        
        # 如果某个类别样本太少，至少保留一个在训练集
        if len(class_images) == 1:
            train_list.extend(class_data.to_dict('records'))
            print(f"  类别 {class_label}: 只有 1 个样本，全部放入训练集")
        else:
            # 分层划分
            train_images, val_images = train_test_split(
                class_images,
                test_size=val_ratio,
                random_state=random_state,
                shuffle=True
            )
            
            train_data = class_data[class_data['Image name'].isin(train_images)]
            val_data = class_data[class_data['Image name'].isin(val_images)]
            
            train_list.extend(train_data.to_dict('records'))
            val_list.extend(val_data.to_dict('records'))
            
            print(f"  类别 {class_label}: 训练集 {len(train_images)} 个, 验证集 {len(val_images)} 个")
    
    # 创建 DataFrame
    df_train_new = pd.DataFrame(train_list)
    df_val = pd.DataFrame(val_list)
    df_test_new = df_test.copy()
    
    # 确保列名一致
    df_train_new = df_train_new[['Image name', 'Retinopathy grade']]
    df_val = df_val[['Image name', 'Retinopathy grade']]
    df_test_new = df_test_new[['Image name', 'Retinopathy grade']]
    
    # 重命名列以便更清晰
    df_train_new.columns = ['filename', 'label']
    df_val.columns = ['filename', 'label']
    df_test_new.columns = ['filename', 'label']
    
    # 显示划分后的统计信息
    print("\n" + "="*60)
    print("划分后的数据集统计:")
    print("="*60)
    print(f"\n训练集: {len(df_train_new)} 个样本")
    print("  类别分布:")
    train_dist = df_train_new['label'].value_counts().sort_index()
    for grade, count in train_dist.items():
        print(f"    类别 {grade}: {count} 个 ({count/len(df_train_new)*100:.2f}%)")
    
    print(f"\n验证集: {len(df_val)} 个样本")
    print("  类别分布:")
    val_dist = df_val['label'].value_counts().sort_index()
    for grade, count in val_dist.items():
        print(f"    类别 {grade}: {count} 个 ({count/len(df_val)*100:.2f}%)")
    
    print(f"\n测试集: {len(df_test_new)} 个样本")
    print("  类别分布:")
    test_dist = df_test_new['label'].value_counts().sort_index()
    for grade, count in test_dist.items():
        print(f"    类别 {grade}: {count} 个 ({count/len(df_test_new)*100:.2f}%)")
    
    # 验证类别比例是否一致
    print("\n" + "="*60)
    print("验证类别比例一致性:")
    print("="*60)
    original_train_dist = df_train['Retinopathy grade'].value_counts(normalize=True).sort_index()
    new_train_dist = df_train_new['label'].value_counts(normalize=True).sort_index()
    val_dist_norm = df_val['label'].value_counts(normalize=True).sort_index()
    
    print("\n原始训练集类别比例 vs 新训练集类别比例 vs 验证集类别比例:")
    for grade in unique_classes:
        orig_ratio = original_train_dist.get(grade, 0) * 100
        new_train_ratio = new_train_dist.get(grade, 0) * 100
        val_ratio_pct = val_dist_norm.get(grade, 0) * 100
        print(f"  类别 {grade}: {orig_ratio:.2f}% -> 训练: {new_train_ratio:.2f}%, 验证: {val_ratio_pct:.2f}%")
    
    # 保存 CSV 文件
    output_dir = dataset_root
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    
    print(f"\n正在保存 CSV 文件...")
    df_train_new.to_csv(train_csv, index=False, encoding='utf-8-sig')
    print(f"  训练集: {train_csv}")
    
    df_val.to_csv(val_csv, index=False, encoding='utf-8-sig')
    print(f"  验证集: {val_csv}")
    
    df_test_new.to_csv(test_csv, index=False, encoding='utf-8-sig')
    print(f"  测试集: {test_csv}")
    
    print("\n处理完成！")
    
    return df_train_new, df_val, df_test_new


if __name__ == "__main__":
    # 配置参数
    DATASET_ROOT = r"I:\Dataset\IDRID\Disease Grading"
    VAL_RATIO = 0.2  # 验证集占原始训练集的 20%
    RANDOM_STATE = 42  # 随机种子，确保结果可复现
    
    # 可选：手动指定标签文件路径（如果自动查找失败）
    TRAIN_LABELS_FILE = 'I:/Dataset/IDRID/Disease Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv'  # 例如: r"I:\Dataset\IDRID\Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
    TEST_LABELS_FILE = 'I:/Dataset/IDRID/Disease Grading/Groundtruths/IDRiD_Disease Grading_Testing Labels.csv'   # 例如: r"I:\Dataset\IDRID\Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv"
    
    try:
        df_train, df_val, df_test = process_idrid_dataset(
            dataset_root=DATASET_ROOT,
            train_labels_file=TRAIN_LABELS_FILE,
            test_labels_file=TEST_LABELS_FILE,
            val_ratio=VAL_RATIO,
            random_state=RANDOM_STATE
        )
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

