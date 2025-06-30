import os
import yaml
import torch
import psutil
from datetime import datetime

def check_memory():
    """检查内存使用情况"""
    memory = psutil.virtual_memory()
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    return {
        'total_memory': memory.total / 1024**3,  # GB
        'available_memory': memory.available / 1024**3,  # GB
        'memory_percent': memory.percent,
        'gpu_memory': gpu_memory
    }

def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """保存配置到YAML文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

def create_default_config():
    """创建默认配置"""
    return {
        'data': {
            'data_path': 'processed_ETH_USDT_data.feather',
            'batch_size': 32,
            'sequence_length': 100,
            'prediction_length': 5,
            'train_ratio': 0.8,
            'num_workers': 4
        },
        'model': {
            'input_dim': 4,
            'sequence_length': 100,
            'prediction_length': 5,
            'd_model': 512,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'nhead': 8,
            'dropout': 0.1,
        },
        'optimizer': {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999],
            'scheduler': 'plateau'
        },
        'training': {
            'epochs': 50,
            'mixed_precision': False
        },
        'distributed': {
            'use_distributed': False,
            'num_workers': 1
        },
        'save_dir': 'checkpoints',
        'resume_from_checkpoint': None
    }

def list_checkpoints(save_dir):
    """列出可用的检查点"""
    checkpoint_dir = os.path.join(save_dir, 'models')
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            file_path = os.path.join(checkpoint_dir, file)
            try:
                # 尝试加载检查点获取信息
                checkpoint = torch.load(file_path, map_location='cpu')
                info = {
                    'path': file_path,
                    'filename': file,
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'step': checkpoint.get('step', 'Unknown'),
                    'best_val_loss': checkpoint.get('best_val_loss', 'Unknown'),
                    'modified_time': os.path.getmtime(file_path)
                }
                checkpoints.append(info)
            except:
                # 如果无法加载，跳过
                continue
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
    return checkpoints