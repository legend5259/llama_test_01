import pandas as pd

# 原文件路径
input_file = "labeled_logs_selected.csv"
output_file = "labeled_logs_selected_duplicated.csv"

# 读取原数据
df = pd.read_csv(input_file)

# 筛选 label = 3 的数据
label_1_rows = df[df['label'] == 3]

# 复制两次（得到 2 份额外副本）
duplicated_rows = pd.concat([label_1_rows] * 2, ignore_index=True)

# 合并原数据和重复数据
new_df = pd.concat([df, duplicated_rows], ignore_index=True)

# 随机打乱顺序
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存到新文件
new_df.to_csv(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")
print(f"原数据行数: {len(df)}, 新数据行数: {len(new_df)}")
