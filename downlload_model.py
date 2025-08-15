# from modelscope.hub.snapshot_download import snapshot_download
# snapshot_download(model_id="Qwen/Qwen2.5-7B-Instruct",cache_dir="F:\my_project\low_jingdu_xunlian\model",ignore_file_pattern=".pth")
# model_id为想要下载模型所对应的id；cache_dir为想把大模型存储到什么位置
# 模型一般会有原始文件和safetensors两个版本，只要一个就行了；ignore_file_pattern=".pth"就会过滤原始文件

from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(model_id="LLM-Research/Meta-Llama-3.1-8B-Instruct",cache_dir="./model")