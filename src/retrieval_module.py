# src/retrieval_module.py

import os
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 初始化日志
logging.basicConfig(level=logging.INFO)


def load_risk_knowledge_base(kb_file: str) -> List[str]:
    """
    加载风险知识库，风险描述存储在 JSON 或 TXT 文件中。
    请根据你的数据格式进行调整。
    示例：kb_file中存放一个列表，每个元素为一个风险点描述
    """
    if not os.path.exists(kb_file):
        logging.error(f"风险知识库文件不存在：{kb_file}")
        return []
    with open(kb_file, 'r', encoding='utf-8') as f:
        # 假设知识库以JSON格式存储，例如：["风险描述1", "风险描述2", ...]
        risk_points = json.load(f)
    logging.info(f"Loaded {len(risk_points)} risk points from {kb_file}")
    return risk_points


def load_paragraphs(processed_dir: str) -> List[Tuple[str, str]]:
    """
    从预处理后的文件中加载合同段落。
    返回值为列表，每个元素为 (file_name, paragraph_text)。
    你需要保证 data/processed 目录下每个txt文件，每行代表一个段落。
    """
    paragraphs = []
    for file in os.listdir(processed_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(processed_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        paragraphs.append((file, line))
    logging.info(f"Loaded {len(paragraphs)} paragraphs from {processed_dir}")
    return paragraphs


def compute_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    使用 SentenceTransformer 模型计算文本嵌入。
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings


def retrieve_risk_paragraphs(paragraphs: List[Tuple[str, str]],
                             risk_points: List[str],
                             model: SentenceTransformer,
                             threshold: float = 0.7) -> Dict[str, List[Dict]]:
    """
    对每个合同段落计算与风险描述的相似度，
    返回一个字典，key为文件名，value为风险召回结果列表，每个结果包含：
       - paragraph: 段落文本
       - risk_point: 匹配的风险描述
       - score: 相似度分数
    threshold 参数用于过滤相似度较低的召回结果。
    """
    # 计算风险描述的嵌入
    risk_embeddings = compute_embeddings(model, risk_points)

    # 按文件组织段落
    file_paragraphs = {}
    for file_name, para in paragraphs:
        file_paragraphs.setdefault(file_name, []).append(para)

    results = {}
    for file_name, paras in file_paragraphs.items():
        # 计算合同段落嵌入
        para_embeddings = compute_embeddings(model, paras)
        # 计算每个段落与所有风险描述的相似度（余弦相似度）
        cosine_scores = util.cos_sim(para_embeddings, risk_embeddings)
        file_results = []
        for idx_para, para in enumerate(paras):
            # 对每个风险描述找到最高得分
            scores = cosine_scores[idx_para].cpu().numpy()
            max_idx = int(np.argmax(scores))
            max_score = scores[max_idx]
            if max_score >= threshold:
                file_results.append({
                    "paragraph": para,
                    "risk_point": risk_points[max_idx],
                    "score": float(max_score)
                })
        results[file_name] = file_results
    return results


if __name__ == "__main__":
    # TODO: 请根据实际情况填写风险知识库文件的路径，例如 "data/risk_knowledge.json"
    kb_file = r"D:\Contract_star\Contract_risk_detection\data\risk_knowledge.json"
    # TODO: 请确保预处理后的合同文本存放目录正确，例如 "data/processed"
    processed_dir = r"D:\Contract_star\Contract_risk_detection\data\processed"

    # 加载风险知识库
    risk_points = load_risk_knowledge_base(kb_file)
    if not risk_points:
        exit(1)

    # 加载合同段落
    paragraphs = load_paragraphs(processed_dir)
    if not paragraphs:
        logging.error("没有加载到任何段落，请检查预处理数据。")
        exit(1)

    # 加载预训练的 SentenceTransformer 模型
    # TODO: 你可以更换为合适的预训练模型，例如 'all-MiniLM-L6-v2'
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    logging.info(f"Loaded SentenceTransformer model: {model_name}")

    # 执行风险召回
    retrieval_results = retrieve_risk_paragraphs(paragraphs, risk_points, model, threshold=0.7)

    # 将结果保存为 JSON 文件，方便后续分析
    output_file = "../experiments/results/risk_retrieval_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved retrieval results to {output_file}")
