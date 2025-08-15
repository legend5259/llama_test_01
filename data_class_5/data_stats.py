import pandas as pd


# 读取CSV文件
df = pd.read_csv("labeled_logs_selected.csv")
# 统计每个label的数量
label_counts = df['label'].value_counts().sort_index()

total_count = len(df)

print(f"总行数: {total_count}")

for label in range(5):
    count = label_counts.get(label, 0)
    ratio = count / total_count
    print(f"label={label} 的数量: {count}，比例: {ratio:.2%}")

df = pd.read_csv("labeled_logs_selected_duplicated.csv")
# 统计每个label的数量
label_counts = df['label'].value_counts().sort_index()

total_count = len(df)

print(f"总行数: {total_count}")

for label in range(5):
    count = label_counts.get(label, 0)
    ratio = count / total_count
    print(f"label={label} 的数量: {count}，比例: {ratio:.2%}")

