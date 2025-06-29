import pandas as pd

# 替换为你的feather文件路径
feather_file = 'OHLCVD_ETH_USDT_data.feather'

# 读取feather文件
df = pd.read_feather(feather_file)

df = df.iloc[:2523]
df.to_feather('OHLCVD_ETH_USDT_data_2523.feather')
# 显示前几行
print(df.head(20))