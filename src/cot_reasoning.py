# src/cot_reasoning_v2.py

import requests
import logging
import time

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# 模型服务配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-uguyxxsgallvomtewqjjhylobterkeokllqgiyehnfjnfpdd"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

def generate_cot_prompt(paragraph: str, risk_point: str) -> str:
    """
    构造 Chain-of-Thought 推理的 Prompt，指导模型逐步分析是否包含风险点。

    参数：
    - paragraph: 合同段落文本
    - risk_point: 风险点描述

    返回：
    - prompt_text: 构造好的 Prompt 字符串
    """
    prompt_text = (
        f"请仔细阅读以下合同段落，并对是否存在风险点进行分步思考：\n"
        f"风险点描述：{risk_point}\n\n"
        f"合同段落：\n\"\"\"\n{paragraph}\n\"\"\"\n\n"
        "请先逐步列出你对合同段落中与风险点相关的观察和分析，每一步请简明扼要地说明理由；\n"
        "然后综合以上分析给出最终结论，格式如下：\n"
        "思考过程：\n"
        "1. ...\n"
        "2. ...\n"
        "...\n"
        "最终结论：包含/不包含\n"
    )
    return prompt_text

def call_llm_with_cot_prompt(prompt_text: str, max_tokens: int = 300, retries: int = 3, timeout: int = 10) -> str:
    """
    调用 LLM 接口执行推理，获取 Chain-of-Thought 输出。

    参数：
    - prompt_text: 输入的 prompt
    - max_tokens: 最大返回 token 数
    - retries: 最大重试次数
    - timeout: 每次请求的超时时间

    返回：
    - response_text: 模型返回的完整文本
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            response_text = response.json()['choices'][0]['message']['content']
            if response_text.strip():
                return response_text
        except Exception as e:
            logging.warning(f"[尝试 {attempt + 1}/{retries}] 调用 LLM 失败: {e}")
            time.sleep(1)

    logging.error("多次尝试后调用大模型失败。")
    return ""

if __name__ == "__main__":
    # 测试运行
    sample_paragraph = "本合同规定，出租方须按时支付租金，若逾期未支付，将收取一定比例的违约金。"
    sample_risk = "逾期支付租金"

    prompt = generate_cot_prompt(sample_paragraph, sample_risk)
    logging.info(f"生成的 Prompt:\n{prompt}")

    response = call_llm_with_cot_prompt(prompt)
    logging.info(f"模型返回：\n{response}")
