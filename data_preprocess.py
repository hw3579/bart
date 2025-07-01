import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_feather("ETH_USDT_5m_20240101_20250630.feather")

# print(df.head())

# 数据预处理
def preprocess_data(df):
    """向量化版本的数据预处理"""
    # 创建新的数据结构
    processed_df = pd.DataFrame()
    
    # 基础特征保持不变
    processed_df['open'] = df['open']
    processed_df['high'] = df['high']
    processed_df['low'] = df['low']
    processed_df['close'] = df['close']
    processed_df['volume'] = df['volume']
    processed_df['delta'] = (df['high'] - df['low']) / df['open'] * 100
    
    # 向量化计算label - 这是关键优化！
    close_prices = df['close'].values
    
    # 使用numpy的向量化操作计算下一个价格的变化率
    current_close = close_prices[:-1]  # 除了最后一个
    next_close = close_prices[1:]      # 除了第一个
    
    # 计算涨跌幅度
    label_values = ((next_close - current_close) / current_close) * 100
    
    # 设置label列
    processed_df['label'] = np.nan
    processed_df.iloc[:-1, processed_df.columns.get_loc('label')] = label_values
    
    # 移除最后一行（没有未来数据）
    processed_df = processed_df.dropna()
    
    return processed_df

# 执行预处理
processed_data = preprocess_data(df)

# 查看结果
print("原始数据形状:", df.shape)
print("预处理后数据形状:", processed_data.shape)
print("\n预处理后数据前20行:")
print(processed_data.head(20))

print("\n数据统计信息:")
print(processed_data.describe())

# 保存预处理后的数据
processed_data.to_feather("OHLCVD_ETH_USDT_5m_20240101_20250630.feather")
print("\n预处理后的数据已保存到: OHLCVD_ETH_USDT_data.feather")

