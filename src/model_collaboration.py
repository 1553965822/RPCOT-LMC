# src/model_collaboration_v2.py

import logging

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def normalize_decision(text: str) -> str:
    """
    提取模型输出中的决策结果，规范化为“包含”或“不包含”。

    参数：
    - text: 模型返回文本

    返回：
    - decision: 标准化判断（包含 / 不包含）
    """
    if not text:
        return ""

    text = text.strip().replace("：", ":")
    for line in text.splitlines()[::-1]:  # 从后向前查找
        if "包含" in line and "不包含" not in line:
            return "包含"
        elif "不包含" in line:
            return "不包含"
    return ""

def combine_decisions(decision_list: list) -> str:
    """
    使用投票机制融合多个模型判断结果。

    参数：
    - decision_list: [{"decision": "包含" or "不包含"}, ...]

    返回：
    - final_decision: "包含" or "不包含"
    """
    vote_count = {"包含": 0, "不包含": 0}
    for entry in decision_list:
        decision = entry.get("decision", "").strip()
        if decision in vote_count:
            vote_count[decision] += 1
        else:
            logging.warning(f"无效决策结果：{decision}")

    if vote_count["包含"] > vote_count["不包含"]:
        return "包含"
    else:
        return "不包含"  # 默认保守处理

def model_collaboration_ensemble(paragraph: str, risk_point: str,
                                 prompt_response: str, cot_response: str) -> dict:
    """
    融合 prompt_engineering 和 cot_reasoning 模型的输出。

    参数：
    - paragraph: 合同段落
    - risk_point: 风险点
    - prompt_response: prompt_engineering 返回文本
    - cot_response: cot_reasoning 返回文本

    返回：
    - result_dict: 包含各模型决策和最终融合结果
    """
    # 提取 Prompt 模型结果
    if "判断结果" in prompt_response:
        prompt_decision = normalize_decision(prompt_response.split("判断结果")[-1])
    else:
        prompt_decision = normalize_decision(prompt_response)

    # 提取 CoT 模型结果
    if "最终结论" in cot_response:
        cot_decision = normalize_decision(cot_response.split("最终结论")[-1])
    else:
        cot_decision = normalize_decision(cot_response)

    decisions = [{"decision": prompt_decision}, {"decision": cot_decision}]
    final_decision = combine_decisions(decisions)

    return {
        "paragraph": paragraph,
        "risk_point": risk_point,
        "prompt_response": prompt_decision,
        "cot_response": cot_decision,
        "final_decision": final_decision
    }

if __name__ == "__main__":
    # 示例
    para = "本合同规定，出租方须按时支付租金，若逾期未支付，将收取一定比例的违约金。"
    risk = "逾期支付租金"
    response1 = "判断结果：包含\n理由：合同明确描述逾期及违约处理。"
    response2 = (
        "思考过程：\n"
        "1. 合同指出租方必须按时付款；\n"
        "2. 描述了逾期的后果，即收取违约金；\n"
        "最终结论：包含"
    )

    fusion_result = model_collaboration_ensemble(para, risk, response1, response2)
    print("融合判断：", fusion_result)
