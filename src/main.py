import logging
from src.datetime import datetime
from src.typing import Dict, List

import yaml
from src.data_preprocessing import ContractProcessor
from src.minicpm_module import MiniCPMFineTuner
from src.retrieval_module import RiskRetriever
from src.utils import save_json_results


def main(config_path: str):
    # 初始化配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # 数据预处理阶段
        logger.info("开始数据预处理")
        processor = ContractProcessor(config['data_config'])
        raw_data = processor.load_data(config['data_config']['raw_path'])
        processed_paragraphs = processor.extract_risk_paragraphs(raw_data)

        # 大模型推理阶段
        logger.info("初始化MiniCPM模型")
        minicpm = MiniCPMFineTuner(config_path)
        risk_analysis_results = []

        # 风险检索增强
        retriever = RiskRetriever(config['retrieval_config'])
        
        for para in processed_paragraphs:
            # 检索增强
            retrieved_context = retriever.retrieve(para)
            augmented_input = f"上下文：{retrieved_context}\n待分析段落：{para}"
            
            # 大模型推理
            analysis_result = minicpm.predict(augmented_input)
            risk_analysis_results.append({
                "paragraph": para,
                "analysis": analysis_result
            })

        # 小模型决策阶段
        logger.info("执行最终风险决策")
        decision_results = []
        from deepseek_module import DeepSeekDecisionModel

        # 初始化决策模型
        decision_model = DeepSeekDecisionModel(config_path)
        
        # 执行风险决策
        for analysis in risk_analysis_results:
            decision = decision_model.predict(
                analysis['paragraph'],
                analysis['analysis']
            )
            decision_results.append({
                'paragraph': analysis['paragraph'],
                'risk_analysis': analysis['analysis'],
                'final_decision': decision
            })
        # 保存结果
        output_path = config['output_config']['result_path'].format(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        # 执行评估
        from evaluation import RiskEvaluator
        evaluator = RiskEvaluator(config['evaluation_config'])
        metrics = evaluator.evaluate(
            decision_results,
            config['data_config']['ground_truth_path']
        )
        save_json_results({
            'results': decision_results,
            'metrics': metrics
        }, output_path)
        logger.info(f"流程执行完成，结果已保存至：{output_path}")

    except FileNotFoundError as e:
        logger.error(f"文件路径错误：{str(e)}")
    except KeyError as e:
        logger.error(f"配置参数错误：{str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='合同风险分析主流程')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)