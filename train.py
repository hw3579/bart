import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import os
import time
import json
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import gc
import psutil
import tyro
from dataclasses import dataclass  # 新增
from typing import Optional  # 新增
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json

# Ray imports for distributed training
try:
    import ray
    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.torch import TorchTrainer
    import ray.train.torch as ray_torch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Training without distribution.")

from bartmodel import create_model, BARTTimeSeriesModel, BARTLoss

from utils import check_memory, load_config, save_config, create_default_config, list_checkpoints

class Trainer:
    def __init__(self, config, use_ray=False):
        """
        训练器类
        
        Args:
            config: 训练配置字典
            use_ray: 是否使用Ray分布式训练
        """
        self.config = config
        self.use_ray = use_ray
        
        # 设置分布式训练环境
        if use_ray and RAY_AVAILABLE:
            self.device = ray_torch.get_device()
            self.local_rank = train.get_context().get_local_rank()
            self.world_size = train.get_context().get_world_size()
            self.is_distributed = True
        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # PyTorch原生分布式
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            
            # 初始化分布式
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.is_distributed = True
        else:
            # 单机训练
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.is_distributed = False
        
        # 检查内存
        if self.local_rank == 0:
            memory_info = check_memory()
            print(f"系统内存: {memory_info['total_memory']:.1f}GB")
            print(f"可用内存: {memory_info['available_memory']:.1f}GB")
            print(f"内存使用率: {memory_info['memory_percent']:.1f}%")
            if memory_info['gpu_memory'] is not None:
                print(f"GPU内存使用: {memory_info['gpu_memory']:.1f}GB")
        
        # 只在主进程设置日志和保存目录
        if self.local_rank == 0:
            self._setup_logging()
            self._setup_save_dir()
        
        # 训练状态 - 先初始化这些，以便加载检查点时使用
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 初始化模型、数据和优化器
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        
        # 加载检查点（如果需要） - 移到最后
        if self.config.get('resume_from_checkpoint'):
            self._load_checkpoint(self.config['resume_from_checkpoint'])
        
        if self.local_rank == 0:
            self.logger.info(f"Training on device: {self.device}")
            self.logger.info(f"World size: {self.world_size}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(self.config.get('save_dir', 'checkpoints'), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_save_dir(self):
        """设置保存目录"""
        self.save_dir = self.config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)
    
    def _setup_model(self):
        """设置模型"""
        model_config = self.config.get('model', {})
        
        # 根据配置选择模型类型
        use_flash_attention = model_config.get('use_flash_attention', False)
        use_xformers = model_config.get('use_xformers', False)
        
        if use_xformers:
            try:
                # 导入xFormers优化版本
                from bartmodel_xformers import create_model
                if self.local_rank == 0:
                    print("Loading xFormers optimized BART model...")
            except ImportError as e:
                # 如果xFormers不可用，回退到标准模型
                if self.local_rank == 0:
                    print(f"xFormers not available ({e}), falling back to standard BART model...")
                from bartmodel import create_model
        elif use_flash_attention:
            try:
                # 导入Flash Attention版本
                from bartmodel_flash_attn import create_model
                if self.local_rank == 0:
                    print("Loading Flash Attention BART model...")
            except ImportError as e:
                # 如果Flash Attention不可用，回退到标准模型
                if self.local_rank == 0:
                    print(f"Flash Attention not available ({e}), using standard BART model...")
                from bartmodel import create_model
        else:
            # 使用标准模型
            from bartmodel import create_model
            if self.local_rank == 0:
                print("Loading standard BART model...")
        
        self.model, self.criterion = create_model(model_config)
        self.model.to(self.device)
        
        # 分布式训练包装
        if self.use_ray and RAY_AVAILABLE:
            self.model = ray_torch.prepare_model(self.model)
        elif self.is_distributed and not self.use_ray:
            # PyTorch原生DDP
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
        
    
    def _setup_data(self):
        """设置数据加载器"""
        data_config = self.config.get('data', {})
        model_config = self.config.get('model', {})  # 添加这行

            
        # 为分布式训练调整batch size
        batch_size = data_config.get('batch_size', 32)
        
        # 根据可用内存调整batch size
        memory_info = check_memory()
        if memory_info['available_memory'] < 8:
            batch_size = min(batch_size, 16)
            print(f"Warning: Reduced batch size to {batch_size} due to low memory")
        
        # 分布式训练时平分batch size
        if self.is_distributed:
            batch_size = batch_size // self.world_size
        
        # 减少数据加载器的工作进程数
        num_workers = min(data_config.get('num_workers', 4), 2)
        
        # 使用dataloader中的create_dataloaders函数
        from dataloader import create_dataloaders
        
        # 获取数据路径
        data_path = data_config.get('data_path', 'processed_ETH_USDT_data.feather')        
        
        # 创建数据加载器（不使用分布式，我们后面会重新包装）
        train_loader_temp, val_loader_temp, self.dataset = create_dataloaders(
            data_path=data_path,
            batch_size=batch_size,
            sequence_length=data_config.get('sequence_length', 200),
            prediction_length=data_config.get('prediction_length', 5),
            train_ratio=data_config.get('train_ratio', 0.8),
            num_workers=0,  # 先设为0，后面重新创建
            feature_columns=model_config.get('feature_columns', ['open', 'delta', 'close', 'volume'])

        )
        
        # 获取数据集用于分布式包装
        # 由于create_dataloaders返回的是包装后的数据集，我们需要获取原始数据集
        train_dataset = train_loader_temp.dataset
        val_dataset = val_loader_temp.dataset
        
        # 创建分布式采样器
        if self.is_distributed and not self.use_ray:
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        # 重新创建数据加载器（带分布式采样器）
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Ray分布式包装
        if self.use_ray and RAY_AVAILABLE:
            self.train_loader = ray_torch.prepare_data_loader(self.train_loader)
            self.val_loader = ray_torch.prepare_data_loader(self.val_loader)
        
        # 保存采样器
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        
        # 只在主进程保存标准化器
        if self.local_rank == 0:
            scaler_path = os.path.join(self.save_dir, 'scalers')
            self.dataset.save_scalers(scaler_path)
            self.logger.info(f"Scalers saved to {scaler_path}")


    def _setup_optimizer(self):
        """设置优化器和调度器"""
        optimizer_config = self.config.get('optimizer', {})
        
        # 为分布式训练调整学习率
        base_lr = optimizer_config.get('lr', 1e-4)
        if self.use_ray and self.world_size > 1:
            # 线性缩放学习率
            lr = base_lr * self.world_size
        else:
            lr = base_lr
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', [0.9, 0.999])
        )
        
        # 分布式训练包装
        if self.use_ray and RAY_AVAILABLE:
            self.optimizer = ray_torch.prepare_optimizer(self.optimizer)
        
        scheduler_type = optimizer_config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5,
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training', {}).get('epochs', 100),
            )
        else:
            self.scheduler = None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        # 设置分布式采样器的epoch
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)
        
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # 只在主进程显示进度条
        if self.local_rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        else:
            progress_bar = self.train_loader
        
        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            try:
                input_seq = input_seq.to(self.device, non_blocking=True)
                target_seq = target_seq.to(self.device, non_blocking=True)
                
                # 前向传播
                self.optimizer.zero_grad()
                predictions = self.model(input_seq)
                loss = self.criterion(predictions, target_seq)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                self.current_step += 1
                
                # 更新进度条（只在主进程）
                if self.local_rank == 0 and hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                    })
                
                # 定期清理内存
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.local_rank == 0:
                        print(f"CUDA out of memory at batch {batch_idx}. Skipping batch.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # 聚合分布式训练的损失
        avg_loss = total_loss / num_batches
        
        if self.is_distributed and not self.use_ray:
            # PyTorch原生分布式损失聚合
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        elif self.use_ray and self.world_size > 1:
            # Ray分布式损失聚合
            avg_loss = ray_torch.all_reduce(torch.tensor(avg_loss)).item() / self.world_size
        
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            if self.local_rank == 0:
                val_iter = tqdm(self.val_loader, desc='Validation')
            else:
                val_iter = self.val_loader
                
            for input_seq, target_seq in val_iter:
                try:
                    input_seq = input_seq.to(self.device, non_blocking=True)
                    target_seq = target_seq.to(self.device, non_blocking=True)
                    
                    predictions = self.model(input_seq)
                    loss = self.criterion(predictions, target_seq)
                    total_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if self.local_rank == 0:
                            print("CUDA out of memory during validation. Skipping batch.")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        # 聚合分布式训练的验证损失
        avg_loss = total_loss / num_batches
        if self.use_ray and self.world_size > 1:
            avg_loss = ray_torch.all_reduce(torch.tensor(avg_loss)).item() / self.world_size
        
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        """主训练循环"""
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        
        if self.local_rank == 0:
            self.logger.info(f"Starting training for {epochs} epochs")
        
        start_time = time.time()
        
        # 修正：从恢复的epoch开始，而不是从0开始
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 日志记录（只在主进程）
            if self.local_rank == 0:
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best_model')
                    self.logger.info(f"New best model saved with val_loss={val_loss:.4f}")
                
                # 每10个epoch保存检查点
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(f'epoch_{epoch}')
            
            # Ray Train报告进度
            if self.use_ray:
                train.report({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss
                })
            
            # 定期清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        total_time = time.time() - start_time
        if self.local_rank == 0:
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            self._save_training_history()
        
        return self.best_val_loss

    def _save_checkpoint(self, name):
        """保存检查点"""
        if self.local_rank != 0:
            return
            
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.save_dir, 'models', f'{name}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Ray Train检查点
        if self.use_ray:
            train.save_checkpoint(Checkpoint.from_dict(checkpoint))
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 处理DDP模型的状态字典
            model_state_dict = checkpoint['model_state_dict']
            
            # 如果当前模型是DDP包装的，但检查点不是，需要添加module前缀
            if isinstance(self.model, DDP):
                if not any(key.startswith('module.') for key in model_state_dict.keys()):
                    model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            else:
                # 如果当前模型不是DDP，但检查点是，需要移除module前缀
                if any(key.startswith('module.') for key in model_state_dict.keys()):
                    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            
            # 加载模型状态
            self.model.load_state_dict(model_state_dict)
            
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 加载训练状态
            self.current_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
            self.current_step = checkpoint.get('step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            if self.local_rank == 0:
                self.logger.info(f"检查点已加载: {checkpoint_path}")
                self.logger.info(f"将从 Epoch {self.current_epoch} 开始训练")
                self.logger.info(f"当前最佳验证损失: {self.best_val_loss:.4f}")
                
        except Exception as e:
            if self.local_rank == 0:
                self.logger.error(f"加载检查点失败: {e}")
            raise e
    
    def _save_checkpoint(self, name):
        """保存检查点"""
        if self.local_rank != 0:
            return
        
        # 获取模型状态字典，处理DDP包装
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.save_dir, 'models', f'{name}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Ray Train检查点
        if self.use_ray:
            train.save_checkpoint(Checkpoint.from_dict(checkpoint))
    
    def _plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.title('Training and Validation Loss (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'), dpi=300)
        plt.close()

    def _save_training_history(self):
        """保存训练历史"""
        if self.local_rank == 0:  # 修改这里，使用 local_rank 而不是 is_main_process
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'epochs_completed': len(self.train_losses)
            }
            
            history_path = os.path.join(self.save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"Training history saved to {history_path}")
            
            # 同时绘制损失曲线
            self._plot_losses()

def train_func(config):
    """Ray分布式训练函数"""
    trainer = Trainer(config, use_ray=True)
    trainer.train()



@dataclass
class TrainingArgs:
    """训练参数配置"""
    config: str = "default.yaml"
    """配置文件路径"""
    
    create_config: bool = False
    """创建默认配置文件"""
    
    resume: Optional[str] = None
    """从检查点恢复训练，指定检查点文件路径"""
    
    list_checkpoints: bool = False
    """列出可用的检查点"""
    
    resume_best: bool = False
    """从最佳模型检查点恢复训练"""
    
    resume_latest: bool = False
    """从最新检查点恢复训练"""



def main():
    """主函数"""
    
    # 设置环境变量避免TensorFlow相关警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    try:
        # 使用tyro解析参数
        args = tyro.cli(TrainingArgs, description="BART Time Series Training")

        # 创建默认配置文件
        if args.create_config:
            default_config = create_default_config()
            save_config(default_config, args.config)
            print(f"默认配置文件已创建: {args.config}")
            return
        
        # 加载配置
        if os.path.exists(args.config):
            config = load_config(args.config)
            print(f"配置文件已加载: {args.config}")
        else:
            print(f"配置文件不存在: {args.config}")
            print("创建默认配置文件...")
            config = create_default_config()
            save_config(config, args.config)
            print(f"默认配置文件已创建: {args.config}")
        
        save_dir = config.get('save_dir', 'checkpoints')
        
        # 列出检查点
        if args.list_checkpoints:
            checkpoints = list_checkpoints(save_dir)
            if checkpoints:
                print("\n可用的检查点:")
                print("-" * 80)
                print(f"{'文件名':<25} {'Epoch':<8} {'Step':<10} {'最佳验证损失':<15} {'修改时间'}")
                print("-" * 80)
                for cp in checkpoints:
                    from datetime import datetime
                    mod_time = datetime.fromtimestamp(cp['modified_time']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{cp['filename']:<25} {str(cp['epoch']):<8} {str(cp['step']):<10} {str(cp['best_val_loss']):<15} {mod_time}")
            else:
                print("没有找到检查点文件")
            return
        
        # 处理resume参数
        resume_path = None
        if args.resume:
            if os.path.exists(args.resume):
                resume_path = args.resume
            else:
                print(f"错误: 检查点文件不存在: {args.resume}")
                return
        elif args.resume_best:
            best_model_path = os.path.join(save_dir, 'models', 'best_model.pth')
            if os.path.exists(best_model_path):
                resume_path = best_model_path
            else:
                print("错误: 未找到最佳模型检查点")
                return
        elif args.resume_latest:
            checkpoints = list_checkpoints(save_dir)
            if checkpoints:
                resume_path = checkpoints[0]['path']  # 最新的检查点
            else:
                print("错误: 未找到任何检查点")
                return
        
        if resume_path:
            config['resume_from_checkpoint'] = resume_path
            print(f"将从检查点恢复训练: {resume_path}")
        
        # 保存当前使用的配置到输出目录
        os.makedirs(save_dir, exist_ok=True)
        save_config(config, os.path.join(save_dir, 'config_used.yaml'))
        
        # 分布式训练配置
        distributed_config = config.get('distributed', {})
        use_distributed = distributed_config.get('use_distributed', False)
        num_workers = distributed_config.get('num_workers', 1)
        
        if use_distributed and RAY_AVAILABLE:
            # 初始化Ray
            ray.init()
            
            # 配置分布式训练
            scaling_config = ScalingConfig(
                num_workers=num_workers,
                use_gpu=True,
                resources_per_worker={"CPU": 2, "GPU": 1}
            )
            
            # 创建TorchTrainer
            trainer = TorchTrainer(
                train_func,
                train_loop_config=config,
                scaling_config=scaling_config,
                run_config=train.RunConfig(
                    name="bart_timeseries_distributed",
                    storage_path="./ray_results",
                    checkpoint_config=train.CheckpointConfig(
                        num_to_keep=5,
                        checkpoint_score_attribute="val_loss",
                        checkpoint_score_order="min",
                    )
                )
            )
            
            # 开始训练
            result = trainer.fit()
            print(f"Training completed. Best checkpoint: {result.checkpoint}")
            
        else:
            # 单机训练
            trainer = Trainer(config, use_ray=False)
            trainer.train()

    finally:
        # 清理分布式进程组
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()