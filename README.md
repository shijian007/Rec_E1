# REC-E1 
Rec-E1 是对 [Rec-R1](https://github.com/linjc16/Rec-R1) 实验的扩展，包含以下升级点：
- 升级了veRL版本，从v0.1 升级为 v0.3
- [todo]升级搜索数据库为ElasticSearch
- 更换了实验数据集，从Rec-R1实验使用的 [Amazon电商数据集](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) ，更换为微软开源的新闻数据集[MIND-small](https://msnews.github.io/)
- 修复了部分install问题


## Installation

```bash
conda create -n zero python=3.10

# verl 当前版本 0.3
pip install -e .

# vllm 当前版本 0.8.4
pip install vllm ray

# flash attention 2, 最好从该项目的release下载whl直接安装，注意选择vllm安装后torch的版本和本地cuda的版本
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib pyserini faiss-gpu

# 最新 pyserini 需要安装 java21 版本
conda install -c conda-forge openjdk=21
export JAVA_HOME=~/miniconda3/envs/zero
export PATH=$JAVA_HOME/bin:$PATH

# 或者 ubuntu 环境下安装jdk21
apt update 
apt install openjdk-21-jdk
# 验证安装:
java --version


```


## 开始

### 数据准备
这里提供了训练集1w条，测试集2k条，及其对应的新闻标题与摘要
```
unzip ./data/mindsmall/mindsmall.zip
# 目录结构
.
+--- train_1w.parquet
+--- dev_2k.parquet
+--- jsonl_docs
|   +--- dev_infocode_docs.jsonl
|   +--- train_infocode_docs.jsonl
```

### 建立 Lucene 索引
```
bash src/Lucene/mindrec/1_build_database.sh
```

### 开始训练
```
conda activate zero
```

如果显存紧张，试着在训练代码中增加，使用部分内存加载模型
`actor_rollout_ref.actor.fsdp_config.param_offload=True`
`actor_rollout_ref.ref.fsdp_config.param_offload=True`

**1.5B model**
```
bash scripts/run_train_qwen_1d5b_mind_small.sh
```


## 致谢
- [Rec_R1](https://github.com/linjc16/Rec-R1) 🔗
- [Verl](https://github.com/volcengine/verl) 🔗
- [Pyserini](https://github.com/castorini/pyserini) 🔗
- [Faiss](https://github.com/facebookresearch/faiss) 🔗