import requests
import logging

# ✅ 模型配置信息
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-xoyefppvnejkgmuylruxfesaqyjjkrzuuiocdupqiazwtfkq"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

def generate_risk_prompt(paragraph: str, risk_point: str) -> str:
    """
    构造用于风险检测任务的 Prompt 模板，指导大模型判断合同段落是否包含风险点。
    """
    prompt = (
        "你是一个合同审查专家，请判断以下合同段落中是否存在指定的风险点。\n\n"
        f"【风险点描述】\n{risk_point}\n\n"
        f"【合同段落】\n\"\"\"\n{paragraph}\n\"\"\"\n\n"
        "请你判断该段落是否明确表达了上述风险点，并严格按照如下格式回答：\n"
        "判断结果：包含 / 不包含\n"
        "理由：请简要说明判断依据，不超过50字。\n"
    )
    return prompt

def call_llm_with_prompt(prompt: str, max_tokens: int = 512) -> str:
    """
    使用大模型 API 对 prompt 进行推理，并返回响应文本。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,  # 可微调以控制生成稳定性
        "max_tokens": max_tokens,
        "response_format": {"type": "text"}
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"调用大模型 API 失败: {e}")
        if response is not None:
            logging.error(f"响应内容: {response.text}")
        return ""

# ✅ 测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_paragraph = (
        "本合同规定，出租方须按时支付租金，若逾期未支付，将收取一定比例的违约金。"
    )
    sample_risk_point = "逾期支付租金"

    prompt = generate_risk_prompt(sample_paragraph, sample_risk_point)
    logging.info("生成的 Prompt:")
    logging.info(prompt)

    result = call_llm_with_prompt(prompt)
    logging.info("大模型返回结果:")
    logging.info(result)
