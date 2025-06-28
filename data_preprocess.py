import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_feather("data/ETH_USDT_USDT-1m-futures.feather")

# print(df.head())

# 数据预处理
def preprocess_data(df):
    # 创建新的数据结构
    processed_df = pd.DataFrame()
    
    # open 保持不变
    processed_df['open'] = df['open']
    
    # delta = high - low
    processed_df['delta'] = df['high'] - df['low']
    
    # close 保持不变
    processed_df['close'] = df['close']
    
    # volume 保持不变
    processed_df['volume'] = df['volume']
    
    # 计算label：后面5根蜡烛的平均值除以当前蜡烛乘100
    # 使用close价格计算
    processed_df['label'] = np.nan
    
    for i in tqdm(range(len(df) - 5), desc="Processing labels"):
        # 计算后面5根蜡烛的平均close价格
        future_5_avg = df['close'].iloc[i+1:i+6].mean()
        current_close = df['close'].iloc[i]
        
        # 计算涨跌幅度 (未来5根平均价格 / 当前价格) * 100
        processed_df.loc[i, 'label'] = ((future_5_avg - current_close) / current_close) * 100

    # 移除最后5行（因为没有足够的未来数据计算label）
    processed_df = processed_df.dropna()
    
    return processed_df

# 执行预处理
processed_data = preprocess_data(df)

# 查看结果
print("原始数据形状:", df.shape)
print("预处理后数据形状:", processed_data.shape)
print("\n预处理后数据前5行:")
print(processed_data.head())

print("\n数据统计信息:")
print(processed_data.describe())

# 保存预处理后的数据
processed_data.to_feather("processed_ETH_USDT_data.feather")
print("\n预处理后的数据已保存到: processed_ETH_USDT_data.feather")

