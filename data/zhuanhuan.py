import csv

# 输入和输出文件路径
input_csv = 'logs.csv'   # 你的原csv文件名
output_csv = 'labeled_logs.csv' # 目标输出文件名

# 定义level到label的映射
level_to_label = {
    'INFO': 0,
    'WARN': 0,
    'DEBUG': 0,
    'ERROR': 1,
    'FATAL': 1
}

with open(input_csv, 'r', encoding='utf-8') as infile, \
     open(output_csv, 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    # 写入表头
    writer.writerow(['label', 'log_text'])

    for row in reader:
        level = row['level'].strip()
        message = row['message'].strip()
        label = level_to_label.get(level, 0)  # 默认0，防止有未知level
        writer.writerow([label, message])
