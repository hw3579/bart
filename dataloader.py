import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, sequence_length=200, prediction_length=5, train=True, train_ratio=0.8, feature_columns=['open', 'delta', 'close', 'volume']):
        """
        Args:
            data_path: 预处理后的数据文件路径
            sequence_length: 输入序列长度 (200根蜡烛)
            prediction_length: 预测序列长度 (5根蜡烛)
            train: 是否为训练集
            train_ratio: 训练集比例
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # 加载数据
        self.data = pd.read_feather(data_path)
        
        # 特征列和标签列
        self.feature_columns = feature_columns
        self.label_column = 'label'  # 使用后5根蜡烛收盘价作为标签
        
        # 数据标准化
        self.scaler_features = StandardScaler()
        self.scaler_labels = StandardScaler()
        
        # 准备特征数据
        features = self.data[self.feature_columns].values
        labels = self.data[self.label_column].values.reshape(-1, 1)
        
        # 拟合标准化器
        self.scaler_features.fit(features)
        self.scaler_labels.fit(labels)
        
        # 标准化数据
        self.features_scaled = self.scaler_features.transform(features)
        self.labels_scaled = self.scaler_labels.transform(labels).flatten()
        
        # 创建序列
        self.sequences = []
        self.targets = []
        
        # 生成所有可能的序列
        for i in range(len(self.data) - sequence_length - prediction_length + 1):
            # 输入序列 (200根蜡烛的特征)
            seq = self.features_scaled[i:i + sequence_length]
            
            # 目标序列 (后5根蜡烛的label)
            target = self.labels_scaled[i + sequence_length:i + sequence_length + prediction_length]
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        # 转换为numpy数组
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        # 划分训练集和验证集
        split_idx = int(len(self.sequences) * train_ratio)
        
        if train:
            self.sequences = self.sequences[:split_idx]
            self.targets = self.targets[:split_idx]
        else:
            self.sequences = self.sequences[split_idx:]
            self.targets = self.targets[split_idx:]
        
        print(f"{'训练集' if train else '验证集'} 大小: {len(self.sequences)}")
        print(f"输入序列形状: {self.sequences.shape}")
        print(f"目标序列形状: {self.targets.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        返回:
            input_seq: (sequence_length, n_features) 输入序列
            target_seq: (prediction_length,) 目标序列
        """
        input_seq = torch.FloatTensor(self.sequences[idx])
        target_seq = torch.FloatTensor(self.targets[idx])
        
        return input_seq, target_seq
    
    def inverse_transform_labels(self, scaled_labels):
        """将标准化后的标签转换回原始尺度"""
        return self.scaler_labels.inverse_transform(scaled_labels.reshape(-1, 1)).flatten()
    
    def save_scalers(self, path_prefix):
        """保存标准化器"""
        with open(f"{path_prefix}_feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler_features, f)
        with open(f"{path_prefix}_label_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler_labels, f)
    
    def load_scalers(self, path_prefix):
        """加载标准化器"""
        with open(f"{path_prefix}_feature_scaler.pkl", 'rb') as f:
            self.scaler_features = pickle.load(f)
        with open(f"{path_prefix}_label_scaler.pkl", 'rb') as f:
            self.scaler_labels = pickle.load(f)

def create_dataloaders(data_path, batch_size=32, sequence_length=200, prediction_length=5, train_ratio=0.8, num_workers=4, feature_columns=['open', 'delta', 'close', 'volume']):
    """
    创建训练和验证数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        sequence_length: 输入序列长度
        prediction_length: 预测序列长度
        train_ratio: 训练集比例
        num_workers: 数据加载器工作进程数
    
    Returns:
        train_loader, val_loader, dataset_train (用于获取标准化器)
    """
    # 创建训练集
    dataset_train = TimeSeriesDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        train=True,
        train_ratio=train_ratio,
        feature_columns=feature_columns
    )
    
    # 创建验证集
    dataset_val = TimeSeriesDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        train=False,
        train_ratio=train_ratio,
        feature_columns=feature_columns 
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset_train

# 测试数据加载器
if __name__ == "__main__":
    # 测试数据加载器
    data_path = "processed_ETH_USDT_data.feather"
    
    train_loader, val_loader, dataset = create_dataloaders(
        data_path=data_path,
        batch_size=16,
        sequence_length=200,
        prediction_length=5
    )
    
    # 保存标准化器
    dataset.save_scalers("models/scalers")
    
    # 测试一个批次
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"输入序列形状: {input_seq.shape}")  # [batch_size, 200, 4]
        print(f"目标序列形状: {target_seq.shape}")  # [batch_size, 5]
        print(f"输入序列范围: [{input_seq.min():.3f}, {input_seq.max():.3f}]")
        print(f"目标序列范围: [{target_seq.min():.3f}, {target_seq.max():.3f}]")
        break
    
    print(f"\n总训练批次数: {len(train_loader)}")
    print(f"总验证批次数: {len(val_loader)}")