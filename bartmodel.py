import torch
import torch.nn as nn
from transformers import BartModel, BartConfig
import math

class BARTTimeSeriesModel(nn.Module):
    def __init__(self, 
                 input_dim=4,  # open, delta, close, volume
                 sequence_length=200,
                 prediction_length=5,
                 d_model=768,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 nhead=12,
                 dropout=0.1,
                 use_flash_attention=False):
        """
        基于BART的时序预测模型
        
        Args:
            input_dim: 输入特征维度
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
            d_model: 模型隐藏维度
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            nhead: 注意力头数
            dropout: dropout概率
        """
        super(BARTTimeSeriesModel, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        
        # BART配置
        config = BartConfig(
            vocab_size=1,  # 不使用词汇表
            d_model=d_model,
            encoder_layers=num_encoder_layers,
            decoder_layers=num_decoder_layers,
            encoder_attention_heads=nhead,
            decoder_attention_heads=nhead,
            encoder_ffn_dim=d_model * 4,
            decoder_ffn_dim=d_model * 4,
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            max_position_embeddings=max(sequence_length, prediction_length) + 10,
            init_std=0.02,
            classifier_dropout=dropout,
            scale_embedding=True,
            use_cache=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        
        # 创建BART模型（不使用预训练权重）
        self.bart = BartModel(config)
        
        # 输入投影层：将时序特征投影到模型维度
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.encoder_pos_embedding = nn.Parameter(
            torch.randn(sequence_length, d_model) * 0.02
        )
        self.decoder_pos_embedding = nn.Parameter(
            torch.randn(prediction_length, d_model) * 0.02
        )
        
        # 解码器输入嵌入
        self.decoder_input_embedding = nn.Parameter(
            torch.randn(prediction_length, d_model) * 0.02
        )
        
        # 输出投影层：将模型输出投影到标量
        self.output_projection = nn.Linear(d_model, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_seq, target_seq=None):
        """
        前向传播
        
        Args:
            input_seq: [batch_size, sequence_length, input_dim] 输入序列
            target_seq: [batch_size, prediction_length] 目标序列（训练时使用）
        
        Returns:
            predictions: [batch_size, prediction_length] 预测结果
        """
        batch_size = input_seq.size(0)
        device = input_seq.device
        
        # 编码器输入处理
        # 投影到模型维度
        encoder_inputs = self.input_projection(input_seq)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        encoder_inputs = encoder_inputs + self.encoder_pos_embedding.unsqueeze(0)
        
        # 调整维度为BART期望的格式
        encoder_inputs = encoder_inputs.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # 解码器输入处理
        # 使用学习的嵌入作为解码器输入
        decoder_inputs = self.decoder_input_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        decoder_inputs = decoder_inputs + self.decoder_pos_embedding.unsqueeze(0)
        decoder_inputs = decoder_inputs.transpose(0, 1)  # [pred_len, batch_size, d_model]
        
        # 创建attention mask
        encoder_attention_mask = torch.ones(
            batch_size, self.sequence_length, device=device, dtype=torch.bool
        )
        
        # BART前向传播
        outputs = self.bart(
            inputs_embeds=encoder_inputs,
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs,
            use_cache=False
        )
        
        # 获取解码器输出
        decoder_outputs = outputs.last_hidden_state  # [pred_len, batch_size, d_model]
        decoder_outputs = decoder_outputs.transpose(0, 1)  # [batch_size, pred_len, d_model]
        
        # 投影到标量输出
        predictions = self.output_projection(decoder_outputs).squeeze(-1)  # [batch_size, pred_len]
        
        return predictions
    
    def generate(self, input_seq, max_length=None):
        """
        生成预测序列（推理时使用）
        
        Args:
            input_seq: [batch_size, sequence_length, input_dim] 输入序列
            max_length: 最大生成长度
        
        Returns:
            predictions: [batch_size, prediction_length] 预测结果
        """
        if max_length is None:
            max_length = self.prediction_length
        
        self.eval()
        with torch.no_grad():
            predictions = self.forward(input_seq)
        
        return predictions

class BARTLoss(nn.Module):
    """BART时序预测损失函数"""
    
    def __init__(self, loss_type='mse', alpha=0.5):
        """
        Args:
            loss_type: 损失函数类型 ('mse', 'mae', 'huber', 'combined')
            alpha: 组合损失中MSE的权重
        """
        super(BARTLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        """
        计算损失
        
        Args:
            predictions: [batch_size, prediction_length] 预测值
            targets: [batch_size, prediction_length] 真实值
        
        Returns:
            loss: 标量损失值
        """
        if self.loss_type == 'mse':
            return self.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            return self.mae_loss(predictions, targets)
        elif self.loss_type == 'huber':
            return self.huber_loss(predictions, targets)
        elif self.loss_type == 'combined':
            mse = self.mse_loss(predictions, targets)
            mae = self.mae_loss(predictions, targets)
            return self.alpha * mse + (1 - self.alpha) * mae
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

def create_model(config=None):
    """
    创建BART时序预测模型
    
    Args:
        config: 模型配置字典
    
    Returns:
        model: BART时序预测模型
        criterion: 损失函数
    """
    if config is None:
        config = {
            'input_dim': 4,
            'sequence_length': 200,
            'prediction_length': 5,
            'd_model': 512,  # 减小模型大小以适应内存
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'nhead': 8,
            'dropout': 0.1
        }
    
    model = BARTTimeSeriesModel(**config)
    criterion = BARTLoss(loss_type='combined', alpha=0.7)
    
    return model, criterion

# 测试模型
if __name__ == "__main__":
    # 创建模型
    model, criterion = create_model()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 测试前向传播
    batch_size = 4
    sequence_length = 200
    input_dim = 4
    prediction_length = 5
    
    # 创建随机输入
    input_seq = torch.randn(batch_size, sequence_length, input_dim)
    target_seq = torch.randn(batch_size, prediction_length)
    
    print(f"\n输入序列形状: {input_seq.shape}")
    print(f"目标序列形状: {target_seq.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        predictions = model(input_seq)
        loss = criterion(predictions, target_seq)
    
    print(f"预测结果形状: {predictions.shape}")
    print(f"损失值: {loss.item():.4f}")
    
    # 测试生成功能
    generated = model.generate(input_seq)
    print(f"生成结果形状: {generated.shape}")