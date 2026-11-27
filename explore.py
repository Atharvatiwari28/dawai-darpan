import pandas as pd


df = pd.read_csv("data/jan_aushadhi_products.csv")


print(df.head())
print(len(df))
print(df.columns)
zero = df[df["MRP"] == 0]
print(len(zero))     

# 6. TODO: kitne items "Surgical & Medical Consumables" group ke hain?
#    (Jawab 319 aana chahiye)

smc = df[df["Group Name"] == "Surgical & Medical Consumables"]
print(len(smc))

# 7. TODO: kitni dawaiyon ka MRP 100 rupaye se ZYADA hai?

hundred = df[df["MRP"] > 100]
print(len(hundred))