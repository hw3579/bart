import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 尝试导入Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available. Please install flash-attn.")
    

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FlashMultiHeadAttention(nn.Module):
    """使用Flash Attention的多头注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.shape
        
        # 线性变换
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为Flash Attention需要的格式: (batch, seqlen, nheads, headdim)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # 使用Flash Attention
        if FLASH_ATTENTION_AVAILABLE and query.device.type == 'cuda':
            try:
                # Flash Attention要求half precision或更高精度
                orig_dtype = q.dtype
                if orig_dtype == torch.float32:
                    q = q.half()
                    k = k.half()
                    v = v.half()
                
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=is_causal,
                )
                
                # 转回原始数据类型
                attn_output = attn_output.to(orig_dtype)
            except Exception as e:
                print(f"Flash Attention failed, falling back to standard attention: {e}")
                attn_output = self._standard_attention(q, k, v, attn_mask, is_causal)
        else:
            # 回退到标准attention
            attn_output = self._standard_attention(q, k, v, attn_mask, is_causal)
        
        # 重塑输出
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, None  # Flash Attention不返回attention weights
    
    def _standard_attention(self, q, k, v, attn_mask=None, is_causal=False):
        """标准attention作为回退"""
        batch_size, seq_len, nhead, head_dim = q.shape
        
        # 重塑为标准attention格式
        q = q.transpose(1, 2)  # (batch, nhead, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if is_causal:
            # 创建因果掩码
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑回Flash Attention格式
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, nhead, head_dim)
        
        return attn_output

class FlashTransformerEncoderLayer(nn.Module):
    """使用Flash Attention的Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = FlashMultiHeadAttention(d_model, nhead, dropout)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, 
                                key_padding_mask=src_key_padding_mask, is_causal=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class FlashTransformerDecoderLayer(nn.Module):
    """使用Flash Attention的Transformer解码器层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = FlashMultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = FlashMultiHeadAttention(d_model, nhead, dropout)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self attention (causal)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                key_padding_mask=tgt_key_padding_mask, is_causal=True)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, 
                                     key_padding_mask=memory_key_padding_mask, is_causal=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class BARTTimeSeriesFlashModel(nn.Module):
    """使用Flash Attention的BART时序预测模型"""
    def __init__(self, 
                 input_dim=4,
                 sequence_length=200,
                 prediction_length=5,
                 d_model=512,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 nhead=8,
                 dropout=0.1,
                 use_flash_attention=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        
        # 输入嵌入
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, 
                                                     max_len=max(sequence_length, prediction_length) + 100)
        
        # Encoder
        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(FlashTransformerEncoderLayer(
                d_model, nhead, d_model * 4, dropout
            ))
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Decoder
        decoder_layers = []
        for _ in range(num_decoder_layers):
            decoder_layers.append(FlashTransformerDecoderLayer(
                d_model, nhead, d_model * 4, dropout
            ))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # 解码器输入嵌入
        self.decoder_input_embedding = nn.Parameter(
            torch.randn(prediction_length, d_model) * 0.02
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, 1)
        
        # Layer normalization
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # 初始化权重
        self._initialize_weights()
        
        if self.use_flash_attention:
            print("Using Flash Attention for BART model acceleration")
        else:
            print("Flash Attention not available, using standard attention")
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, src):
        """
        前向传播
        
        Args:
            src: [batch_size, sequence_length, input_dim] 输入序列
        
        Returns:
            predictions: [batch_size, prediction_length] 预测结果
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码器输入处理
        src = self.input_projection(src)  # [batch_size, seq_len, d_model]
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        src = self.positional_encoding(src)
        
        # 编码器前向传播
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory.transpose(0, 1)).transpose(0, 1)
        
        memory = self.encoder_norm(memory)
        memory = memory.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 解码器输入处理
        tgt = self.decoder_input_embedding.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, pred_len, d_model]
        tgt = tgt.transpose(0, 1)  # [pred_len, batch_size, d_model]
        tgt = self.positional_encoding(tgt)
        
        # 解码器前向传播
        output = tgt
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(
                output.transpose(0, 1), 
                memory
            ).transpose(0, 1)
        
        output = self.decoder_norm(output)
        output = output.transpose(0, 1)  # [batch_size, pred_len, d_model]
        
        # 输出投影
        predictions = self.output_projection(output).squeeze(-1)  # [batch_size, pred_len]
        
        return predictions

class BARTLoss(nn.Module):
    """BART时序预测损失函数"""
    
    def __init__(self, loss_type='combined', alpha=0.7):
        super().__init__()
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
    创建Flash Attention BART时序预测模型
    
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
            'd_model': 512,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'nhead': 8,
            'dropout': 0.1,
            'use_flash_attention': True
        }
    
    # 确保use_flash_attention参数存在
    config['use_flash_attention'] = config.get('use_flash_attention', True)
    
    model = BARTTimeSeriesFlashModel(**config)
    criterion = BARTLoss(loss_type='combined', alpha=0.7)
    
    return model, criterion

# 测试模型
if __name__ == "__main__":
    # 测试Flash Attention是否可用
    print(f"Flash Attention Available: {FLASH_ATTENTION_AVAILABLE}")
    
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
    
    # 测试CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        input_seq = input_seq.cuda()
        target_seq = target_seq.cuda()
        print("Using CUDA")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        predictions = model(input_seq)
        loss = criterion(predictions, target_seq)
    
    print(f"预测结果形状: {predictions.shape}")
    print(f"损失值: {loss.item():.4f}")