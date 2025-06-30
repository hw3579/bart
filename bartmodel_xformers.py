import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xFormers not available. Using standard attention.")

class XFormersMultiHeadAttention(nn.Module):
    """使用xFormers优化的多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, is_cross_attention: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_cross_attention = is_cross_attention
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Query, Key, Value projections
        if is_cross_attention:
            # Cross attention: Q from decoder, K&V from encoder
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False) 
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            # Self attention: combined QKV projection for efficiency
            self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, tgt_len, _ = query.size()
        
        if self.is_cross_attention:
            # Cross attention
            assert key is not None and value is not None
            src_len = key.size(1)
            
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            # Self attention
            if key is None:
                key = query
            if value is None:
                value = query
                
            src_len = tgt_len
            qkv = self.qkv_proj(query)
            q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim)
        
        if XFORMERS_AVAILABLE:
            # 使用xFormers的memory efficient attention
            attn_bias = None
            
            # 处理attention mask
            if attn_mask is not None or key_padding_mask is not None:
                # 创建bias tensor
                attn_bias = torch.zeros(
                    batch_size, self.num_heads, tgt_len, src_len,
                    device=query.device, dtype=query.dtype
                )
                
                # 应用key padding mask
                if key_padding_mask is not None:
                    # key_padding_mask: [batch, src_len] -> expand to [batch, heads, tgt_len, src_len]
                    expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(
                        batch_size, self.num_heads, tgt_len, src_len
                    )
                    attn_bias = attn_bias.masked_fill(expanded_mask, float('-inf'))
                
                # 应用attention mask (通常用于causal masking)
                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        # [tgt_len, src_len] -> [batch, heads, tgt_len, src_len]
                        attn_bias = attn_bias + attn_mask.unsqueeze(0).unsqueeze(0)
                    elif attn_mask.dim() == 4:
                        # [batch, heads, tgt_len, src_len]
                        attn_bias = attn_bias + attn_mask
            
            # xFormers memory efficient attention
            attn_output = memory_efficient_attention(
                query=q,
                key=k,
                value=v,
                attn_bias=attn_bias,
                p=self.dropout_p if self.training else 0.0,
                scale=self.scale
            )
            
        else:
            # 标准注意力实现作为回退
            q = q.transpose(1, 2)  # [batch, heads, tgt_len, head_dim]
            k = k.transpose(1, 2)  # [batch, heads, src_len, head_dim]
            v = v.transpose(1, 2)  # [batch, heads, src_len, head_dim]
            
            # 计算注意力分数
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # 应用mask
            if attn_mask is not None:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            
            attn_weights = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn_weights = torch.dropout(attn_weights, self.dropout_p, True)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2)  # [batch, tgt_len, heads, head_dim]
        
        # 重塑回原始形状
        attn_output = attn_output.contiguous().view(batch_size, tgt_len, self.d_model)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output, None  # 保持与原始接口一致

class XFormersTransformerEncoderLayer(nn.Module):
    """使用xFormers优化的Transformer编码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = XFormersMultiHeadAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            is_cross_attention=False
        )
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self attention block
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class XFormersTransformerDecoderLayer(nn.Module):
    """使用xFormers优化的Transformer解码器层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = XFormersMultiHeadAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            is_cross_attention=False
        )
        
        # Cross attention
        self.cross_attn = XFormersMultiHeadAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            is_cross_attention=True
        )
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self attention block
        tgt2, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention block
        tgt2, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class XFormersBARTTimeSeriesModel(nn.Module):
    """使用xFormers优化的BART时序预测模型"""
    
    def __init__(self, 
                 input_dim=4,
                 sequence_length=200,
                 prediction_length=5,
                 d_model=768,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 nhead=12,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        
        # 输入投影层
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
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            XFormersTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(num_encoder_layers)
        ])
        
        # Transformer解码器
        self.decoder_layers = nn.ModuleList([
            XFormersTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(num_decoder_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, 1)
        
        # Layer norm
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_seq: torch.Tensor, target_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        encoder_inputs = self.input_projection(input_seq)  # [batch_size, seq_len, d_model]
        encoder_inputs = encoder_inputs + self.encoder_pos_embedding.unsqueeze(0)
        
        # 编码器前向传播
        encoder_output = encoder_inputs
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        encoder_output = self.encoder_norm(encoder_output)
        
        # 解码器输入处理
        decoder_inputs = self.decoder_input_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        decoder_inputs = decoder_inputs + self.decoder_pos_embedding.unsqueeze(0)
        
        # 创建causal mask for decoder self-attention
        tgt_mask = torch.triu(
            torch.ones(self.prediction_length, self.prediction_length, device=device) * float('-inf'),
            diagonal=1
        )
        
        # 解码器前向传播
        decoder_output = decoder_inputs
        for layer in self.decoder_layers:
            decoder_output = layer(
                tgt=decoder_output,
                memory=encoder_output,
                tgt_mask=tgt_mask
            )
        decoder_output = self.decoder_norm(decoder_output)
        
        # 输出投影
        predictions = self.output_projection(decoder_output).squeeze(-1)  # [batch_size, pred_len]
        
        return predictions
    
    def generate(self, input_seq: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
        """
        生成预测序列（推理时使用）
        """
        if max_length is None:
            max_length = self.prediction_length
        
        self.eval()
        with torch.no_grad():
            predictions = self.forward(input_seq)
        
        return predictions

# 损失函数保持不变
class BARTLoss(nn.Module):
    """BART时序预测损失函数"""
    
    def __init__(self, loss_type='mse', alpha=0.5):
        super(BARTLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
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
    创建xFormers优化的BART时序预测模型
    
    Args:
        config: 模型配置字典
    
    Returns:
        model: xFormers优化的BART时序预测模型
        criterion: 损失函数
    """
    if config is None:
        config = {
            'input_dim': 4,
            'sequence_length': 200,
            'prediction_length': 5,
            'd_model': 512,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'nhead': 8,
            'dropout': 0.1,
        }
    
    # 过滤掉不需要的配置项
    model_config = {k: v for k, v in config.items() if k != 'feature_columns'}
    
    model = XFormersBARTTimeSeriesModel(**model_config)
    criterion = BARTLoss(loss_type='combined', alpha=0.7)
    
    return model, criterion

# 测试模型
if __name__ == "__main__":
    # 测试xFormers优化模型
    print(f"xFormers available: {XFORMERS_AVAILABLE}")
    
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