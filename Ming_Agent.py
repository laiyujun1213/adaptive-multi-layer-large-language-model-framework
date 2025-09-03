import json
import re
from openai import OpenAI

# 初始化DeepSeek客户端（使用提供的API Key）
deepseek_client = OpenAI(
    api_key="sk-c129887e59be45e8be3d5a8761fc0392",
    base_url="https://api.deepseek.com"
)

def decompose_construction_task():
    # 1. 定义待分解的建筑类笼统任务（参考文档中复杂任务场景）
    construction_task = "我想要去饮水机接一杯热水，之后去泡一杯咖啡"
    
    # 2. 构造提示词，要求分解为子任务（遵循文档中主代理任务分解逻辑）
    prompt = f"""
    请将以下现实生活中的任务分解为多个具体可执行的子任务，要求逻辑连贯、步骤合理，覆盖前期准备、实施及校验环节：
    任务：{construction_task}
    输出格式要求：以编号列表形式返回子任务，每个子任务用简洁自然语言描述，例如：
    1. 子任务1描述
    2. 子任务2描述
    ...
    """
    
    # 3. 调用DeepSeek API进行任务分解
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # 假设使用DeepSeek的对话模型
            messages=[
                {"role": "system", "content": "你是任务分解专家，擅长将复杂的任务拆解为可执行子任务。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # 降低随机性，确保分解结果更稳定
        )
        
        # 4. 提取并解析API返回结果
        decomposition_result = response.choices[0].message.content
        
        # 使用正则提取编号列表形式的子任务
        subtasks = re.findall(r'\d+\. (.*?)(?=\n\d+\. |$)', decomposition_result, re.DOTALL)
        
        # 5. 构造规范化JSON数据
        result_json = {
            "original_task": construction_task,
            "subtasks": subtasks,
            "decomposition_method": "主代理(DeepSeek API)动态分解",
            "total_subtasks": len(subtasks)
        }
        
        # 6. 保存为JSON文件
        with open("主代理任务分解.json", "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        
        print("任务分解完成，结果已保存至：主代理任务分解.json")
        return result_json
    
    except Exception as e:
        print(f"任务分解失败：{str(e)}")
        return None

# 执行任务分解
if __name__ == "__main__":
    decompose_construction_task()