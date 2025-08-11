import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 文件路径
standard_file = './predictions_with_label.csv'
predicted_file = './predictions_with_label.csv'

# 读取文件
df_truth = pd.read_csv(standard_file)
df_pred = pd.read_csv(predicted_file)

# 检查行数是否一致
if len(df_truth) != len(df_pred):
    raise ValueError("两个文件的行数不一致，无法一一对应比较")

# 提取真实标签和预测标签
labels = df_truth['label']
preds = df_pred['pred_label']

# 计算指标
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
accuracy = accuracy_score(labels, preds)

# 输出结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
