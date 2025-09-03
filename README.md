# adaptive-multi-layer-large-language-model-framework
This is an adaptive multi-layer large language model framework project framework, an open source project for task decomposition in the field of artificial intelligence

## 📖 Project Overview

This project implements a hierarchical task decomposition framework that progressively breaks down complex construction tasks into concrete, executable steps through the collaborative work of a **Main Agent** and **Subagents**.

### ✨ Core Functionality

- The principal agent decomposes the original task into subtasks.
- Sub-agents further decompose sub-tasks into fine-grained steps.
- Automatically match atomic actions with environmental objects
- Dynamic Update of Action Library and Item Library
- Generate normalized task decomposition results
- Multi-Version Evaluation Script Supports Task Decomposition Quality Analysis

## 🛠️ Environment Configuration

### Prerequisites

- Python 3.8+
- Anaconda or Miniconda environment
- Valid DeepSeek API key (requires separate application)
- `all-MiniLM-L6-v2`Pre-trained models (must be downloaded from official channels or the Mota Community)
- https://modelscope.cn/models/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### Model Download and Placement

1. Download the `all-MiniLM-L6-v2` model file
2. Extract the downloaded model to the project root directory, ensuring the path is: `./all-MiniLM-L6-v2` (The evaluation script will load the model from this path by default).

## Clone Repository
```
git clone https://github.com/laiyujun1213/Building-Automation.git
cd Building-Automation
```

## Conda Virtual Environment Setup and Dependency Installation

## Create and activate the virtual environment

### Create a Conda virtual environment
```
conda create -n construction python=3.8 -y
```

### Activate the virtual environment
```
conda activate construction
```

### Install dependencies
```
pip install -r requirements.txt
```

## Configure API Key
Before running the project, replace the DeepSeek API key in the code:
Open the Ming_Agent.py file and replace the value of the api_key parameter.
Open the subagent2.py file and replace the value of the api_key parameter.
#### Example (in Ming_Agent.py and subagent2.py)
```
deepseek_client = OpenAI(
    api_key="你的DeepSeek API密钥",  # 替换为实际密钥
    base_url="https://api.deepseek.com"
)
```

### Modify task content (optional)
To break down custom construction tasks, modify the `construction_task` variable in `Ming_Agent.py`:
#### In Ming_Agent.py

```
construction_task = "你的建筑施工任务描述"  # 例如："规划建筑施工场地的临时设施布置..."
```


## 📋Operate the project
### Launch the entire task decomposition process via the main entry script app.py:
```
python3 app.py
```
First, execute Ming_Agent.py for main task decomposition to generate 主代理任务分解.json.
Next, execute subagent2.py for fine-grained decomposition to generate 细粒度任务分解结果.json.
Then, automatically update construction_environment_items.json (environment item library) and construction_atomic_actions.json (atomic action library).
Finally, execute the evaluation script (using evaluate_decomposition2.py by default) to generate 分解评估结果.json.

### Execute the project in modules
Launch the entire task decomposition process via the main entry script logic (ensure the Conda environment is activated):
#### First, run the main proxy to decompose the task.
```
python3 Ming_Agent.py
```

#### Then run sub-proxies for fine-grained decomposition.
```
python3 subagent2.py
```

#### Finally, run the evaluation script (using the Enhanced Edition as an example).
```
python3 evaluate_decomposition2.py
```



## 📁 Project Structure

```
├── Ming_Agent.py                  # 主代理模块，负责初始任务分解
├── subagent2.py                   # 核心子代理模块，负责细粒度步骤分解（调用API）
├── subagent1.py                   # 辅助子模块，生成动作-物品执行描述
├── app.py                         # 项目主入口，协调各模块执行
├── evaluate_decomposition.py      # 基础版评估脚本
├── evaluate_decomposition1.py     # 优化版评估脚本
├── evaluate_decomposition2.py     # 增强版评估脚本
├── requirements.txt               # 项目依赖列表
├── construction_environment_items.json  # 环境物品库（自动更新）
├── construction_atomic_actions.json     # 原子动作库（自动更新）
├── 主代理任务分解.json              # 主代理分解结果（运行后生成）
├── 细粒度任务分解结果.json          # 细粒度分解结果（运行后生成）
├── 分解评估结果.json                # 评估结果（运行后生成）
└── all-MiniLM-L6-v2/              # 预训练模型目录（需自行下载）
```
















































