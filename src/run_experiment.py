import os
import json
import logging
from datetime import datetime
from retrieval_module import load_risk_knowledge_base, load_paragraphs, retrieve_risk_paragraphs
from prompt_engineering import generate_risk_prompt, call_llm_with_prompt
from cot_reasoning import generate_cot_prompt, call_llm_with_cot_prompt
from model_collaboration import model_collaboration_ensemble
from evaluation import evaluate_metrics

from sentence_transformers import SentenceTransformer, util

# 日志与结果保存路径
EXPERIMENT_DIR = r"D:\Contract_star\Contract_risk_detection\experiments"
RESULT_DIR = os.path.join(EXPERIMENT_DIR, "results")
LOG_DIR = os.path.join(EXPERIMENT_DIR, "logs")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 初始化日志文件
log_file = os.path.join(LOG_DIR, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_ground_truth(gt_file: str) -> dict:
    if not os.path.exists(gt_file):
        logging.error(f"Ground truth 文件不存在: {gt_file}")
        return {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_experiment():
    # ✅ 参数路径配置
    kb_file = r"D:\Contract_star\Contract_risk_detection\data\risk_knowledge.json"
    processed_dir = r"D:\Contract_star\Contract_risk_detection\data\processed"
    ground_truth_file = r"D:\Contract_star\Contract_risk_detection\data\ground_truth.json"

    # ✅ 嵌入模型参数（可改）
    embed_model_name = "paraphrase-MiniLM-L6-v2"
    sim_threshold = 0.65  # ✅ 相似度阈值

    try:
        model = SentenceTransformer(embed_model_name)
        logging.info(f"嵌入模型 [{embed_model_name}] 加载成功。")
    except Exception as e:
        logging.error(f"嵌入模型加载失败: {e}")
        return

    risk_points = load_risk_knowledge_base(kb_file)
    paragraphs = load_paragraphs(processed_dir)
    retrieval_results = retrieve_risk_paragraphs(paragraphs, risk_points, model=model)
    ground_truth = load_ground_truth(ground_truth_file)

    all_true_labels = []
    all_pred_labels = []

    for file_name, risk_results in retrieval_results.items():
        logging.info(f"处理文件: {file_name}, 风险召回条数: {len(risk_results)}")
        gt_items = ground_truth.get(file_name, [])

        for item in risk_results:
            paragraph = item["paragraph"]
            risk_point = item["risk_point"]

            # Prompt + LLM
            prompt = generate_risk_prompt(paragraph, risk_point)
            prompt_response = call_llm_with_prompt(prompt)

            cot_prompt = generate_cot_prompt(paragraph, risk_point)
            cot_response = call_llm_with_cot_prompt(cot_prompt)

            ensemble_result = model_collaboration_ensemble(paragraph, risk_point, prompt_response, cot_response)
            final_decision = ensemble_result["final_decision"]
            logging.info(f"文件: {file_name}, 段落: {paragraph[:30]}..., 风险点: {risk_point}, 融合结果: {final_decision}")

            # GT 匹配
            para_emb = model.encode(paragraph, convert_to_tensor=True)
            best_score = 0.0
            gt_label = None

            for gt in gt_items:
                if gt["risk_point"] != risk_point:
                    continue
                gt_para_emb = model.encode(gt["paragraph"], convert_to_tensor=True)
                sim = util.cos_sim(para_emb, gt_para_emb).item()
                if sim > best_score and sim > sim_threshold:
                    best_score = sim
                    gt_label = gt["label"]

            if gt_label is None:
                logging.warning(f"未在 ground truth 中匹配到文件 {file_name} 的风险段落，跳过。")
                continue

            all_true_labels.append(gt_label)
            all_pred_labels.append(final_decision)

    if all_true_labels and all_pred_labels:
        metrics = evaluate_metrics(all_true_labels, all_pred_labels)
        logging.info("实验评估指标:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.4f}")

        # ✅ 保存结果
        result_path = os.path.join(RESULT_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info(f"评估结果已保存至: {result_path}")
    else:
        logging.warning("未收集到有效评估数据，评估指标无法生成。")

if __name__ == "__main__":
    run_experiment()
