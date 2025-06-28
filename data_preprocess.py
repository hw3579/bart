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
    
    # 保留 high 和 low
    processed_df['high'] = df['high']
    processed_df['low'] = df['low']
    
    # close 保持不变
    processed_df['close'] = df['close']
    
    # volume 保持不变
    processed_df['volume'] = df['volume']

    # delta = high - low
    processed_df['delta'] = (df['high'] - df['low'])/ df['open'] * 100 # 计算波动率，单位为百分比
    
    # 计算label：后面5根蜡烛的平均值除以当前蜡烛乘100
    # 使用close价格计算
    processed_df['label'] = np.nan
    
    for i in tqdm(range(len(df) - 1), desc="Processing labels"):
        # 计算当前蜡烛对于往前第1个的变化
        five_close = df['close'].iloc[i+1]
        current_close = df['close'].iloc[i]
        
        # 计算涨跌幅度 (未来5根平均价格 / 当前价格) * 100
        processed_df.loc[i, 'label'] = ((five_close - current_close) / current_close) * 100

    # 移除最后5行（因为没有足够的未来数据计算label）
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
processed_data.to_feather("OHLCVD_ETH_USDT_data.feather")
print("\n预处理后的数据已保存到: OHLCVD_ETH_USDT_data.feather")

