import os
import pandas as pd
import random
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

def select_diverse_logs(input_csv: str, output_csv: str, target_count: int = 3000):
    # 读取原始数据
    df = pd.read_csv(input_csv)
    if 'log_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("输入 CSV 必须包含 'label' 和 'log_text' 两列")

    # 初始化 Drain3（无持久化）
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = 0.4
    config.drain_max_children = 100
    config.drain_depth = 4
    config.drain_max_clusters = 10000
    template_miner = TemplateMiner(None, config)

    # 提取模板
    log_template_map = []
    for idx, log in enumerate(df['log_text']):
        result = template_miner.add_log_message(str(log))
        template_id = result["cluster_id"]
        log_template_map.append((idx, template_id))

    # 按模板分组
    template_to_logs = {}
    for idx, tid in log_template_map:
        template_to_logs.setdefault(tid, []).append(idx)

    print(f"共提取到 {len(template_to_logs)} 个模板")

    # 先保证不同模板
    selected_indices = []
    for tid, idx_list in template_to_logs.items():
        selected_indices.append(random.choice(idx_list))
        if len(selected_indices) >= target_count:
            break

    # 如果还不够，就随机补齐
    if len(selected_indices) < target_count:
        all_remaining = [
            i for idx_list in template_to_logs.values()
            for i in idx_list if i not in selected_indices
        ]
        random.shuffle(all_remaining)
        selected_indices.extend(all_remaining[:target_count - len(selected_indices)])

    # 直接提取 label 和 log_text
    records = []
    for idx in sorted(set(selected_indices)):
        label = df.iloc[idx]['label']
        log_text = df.iloc[idx]['log_text']
        records.append({
            "label": label,
            "log_text": log_text
        })

    out_df = pd.DataFrame(records)
    output_path = os.path.join(os.getcwd(), output_csv)
    out_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"已生成 {len(out_df)} 条样本，保存到 {output_path}")


if __name__ == '__main__':
    select_diverse_logs("labeled_logs.csv", "labeled_logs_selected.csv", target_count=3000)
