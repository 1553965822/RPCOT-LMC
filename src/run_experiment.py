import os
import json
import time
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, ConfigManager, get_absolute_path, create_directory_if_not_exists
from src.data_preprocessing import ContractPreprocessor
from src.retrieval_module import RiskKnowledgeRetriever
from src.deepseek_module import DeepSeekAnalyzer
from src.minicpm_module import MiniCPMAnalyzer
from src.fine_tuning import MiniCPMFineTuner
from src.model_collaboration import ModelCollaborationManager
from src.cot_reasoning import ChainOfThoughtReasoner
from src.prompt_engineering import PromptEngineer
from src.evaluation import RiskAnalysisEvaluator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Risk Analysis for Contract Documents")
    
    parser.add_argument("--contract_file", "-f", type=str, default=None,
                        help="Path to contract file for analysis")
    parser.add_argument("--risk_point", "-r", type=str, default=None,
                        help="Specific risk point to analyze")
    parser.add_argument("--model", "-m", type=str, default="collaboration",
                        choices=["deepseek", "minicpm", "collaboration", "cot", "all"],
                        help="Model to use for analysis")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Directory for output results")
    parser.add_argument("--fine_tune", "-ft", action="store_true",
                        help="Run fine-tuning on MiniCPM model before analysis")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()

class RiskAnalysisExperiment:
    """Main class for running risk analysis experiments"""
    
    def __init__(self, args=None):
        """Initialize the experiment with configuration"""
        # Load configuration
        self.config = ConfigManager()
        
        # Initialize components
        self.preprocessor = ContractPreprocessor()
        self.risk_retriever = RiskKnowledgeRetriever()
        self.deepseek_analyzer = DeepSeekAnalyzer()
        self.minicpm_analyzer = MiniCPMAnalyzer()
        self.model_collaborator = ModelCollaborationManager()
        self.cot_reasoner = ChainOfThoughtReasoner()
        self.prompt_engineer = PromptEngineer()
        self.evaluator = RiskAnalysisEvaluator()
        
        # Set up from arguments if provided
        if args:
            self.contract_file = args.contract_file or self.config.data_config.get('test_contract_file', None)
            self.risk_point = args.risk_point
            self.model = args.model
            self.output_dir = args.output_dir or self.config.experiment_config.get('results_dir', 'experiments/results')
            self.fine_tune = args.fine_tune
            self.debug = args.debug
        else:
            self.contract_file = self.config.data_config.get('test_contract_file', None)
            self.risk_point = None
            self.model = "collaboration"
            self.output_dir = self.config.experiment_config.get('results_dir', 'experiments/results')
            self.fine_tune = False
            self.debug = False
        
        # Ensure output directory exists
        self.output_dir = get_absolute_path(self.output_dir)
        create_directory_if_not_exists(self.output_dir)
        
        # Set up experiment ID
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def run_fine_tuning(self):
        """Run fine-tuning for MiniCPM model if requested"""
        if self.fine_tune:
            logger.info("Starting MiniCPM fine-tuning process")
            fine_tuner = MiniCPMFineTuner()
            results = fine_tuner.run_fine_tuning_pipeline()
            
            if results["success"]:
                logger.info(f"Fine-tuning completed successfully. Model accuracy: {results['metrics']['accuracy']:.4f}")
                return True
            else:
                logger.error(f"Fine-tuning failed: {results.get('error', 'Unknown error')}")
                return False
        
        return True  # Return True if no fine-tuning was requested
    
    def run_experiment(self):
        """Run the main experiment workflow"""
        try:
            # Step 0: Fine-tune MiniCPM if requested
            if self.fine_tune:
                fine_tuning_success = self.run_fine_tuning()
                if not fine_tuning_success:
                    logger.error("Aborting experiment due to fine-tuning failure")
                    return None
            
            # Step 1: Process contract file
            logger.info(f"Starting experiment {self.experiment_id}")
            
            if self.contract_file:
                logger.info(f"Processing contract file: {self.contract_file}")
                paragraphs = self.preprocessor.extract_paragraphs_from_file(self.contract_file)
            else:
                # Use sample paragraphs for testing if no file provided
                logger.info("No contract file provided, using sample paragraphs")
                paragraphs = [
                    "甲方可随时解除合同，无需提前通知乙方，也无需支付任何补偿。",
                    "乙方应在本合同终止后30天内移交所有工作资料，如有延迟，每天向甲方支付合同总金额的5%作为违约金。",
                    "双方一致同意，任何纠纷应通过友好协商解决，协商不成的，任何一方可向有管辖权的人民法院提起诉讼。"
                ]
                
            logger.info(f"Extracted {len(paragraphs)} paragraphs")
            
            # Step 2: Determine risk points to analyze
            if self.risk_point:
                risk_points = [self.risk_point]
            else:
                # Use default risk points from knowledge base
                risk_points = self.risk_retriever.get_risk_points()
                if not risk_points:
                    risk_points = ["合同解除风险", "违约责任风险", "争议解决风险"]
                    
            logger.info(f"Will analyze for risk points: {', '.join(risk_points)}")
            
            # Step 3: Run analysis for each risk point
            all_results = {}
            
            for risk_point in risk_points:
                logger.info(f"Analyzing for risk point: {risk_point}")
                
                # Run analysis based on selected model
                if self.model == "deepseek":
                    results = self._run_deepseek_analysis(paragraphs, risk_point)
                elif self.model == "minicpm":
                    results = self._run_minicpm_analysis(paragraphs, risk_point)
                elif self.model == "cot":
                    results = self._run_cot_analysis(paragraphs, risk_point)
                elif self.model == "all":
                    results = self._run_all_models_analysis(paragraphs, risk_point)
                else:  # Default to collaboration
                    results = self._run_collaboration_analysis(paragraphs, risk_point)
                
                all_results[risk_point] = results
                
                # Calculate and save evaluation metrics
                metrics = self.evaluator.calculate_metrics(results)
                experiment_name = f"{self.experiment_id}_{risk_point.replace(' ', '_')}"
                self.evaluator.save_evaluation_results(metrics, results, experiment_name)
                
                # Generate and print evaluation report
                report = self.evaluator.generate_evaluation_report(metrics, experiment_name)
                logger.info(f"Evaluation report for {risk_point}:")
                logger.info(report)
            
            # Step 4: Save combined results
            self._save_combined_results(all_results)
            
            logger.info(f"Experiment {self.experiment_id} completed successfully")
            return all_results
            
        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _run_deepseek_analysis(self, paragraphs: List[str], risk_point: str) -> List[Dict[str, Any]]:
        """Run analysis using DeepSeek model with CoT reasoning"""
        logger.info("Running analysis with DeepSeek model (large model)")
        results = []
        
        # First, retrieve relevant risk knowledge for the risk point
        risk_knowledge = self.risk_retriever.get_risk_knowledge(risk_point)
        logger.info(f"Retrieved risk knowledge for {risk_point}")
        
        for i, paragraph in enumerate(paragraphs):
            if self.debug:
                logger.info(f"Analyzing paragraph {i+1}/{len(paragraphs)}")
            
            # Generate CoT prompts with risk knowledge context
            cot_prompt = self.prompt_engineer.generate_cot_prompt(paragraph, risk_point, risk_knowledge)
            
            # Perform CoT reasoning using DeepSeek
            cot_result = self.cot_reasoner.perform_cot_analysis(paragraph, risk_point, cot_prompt)
            
            # Get the final decision from DeepSeek
            result = self.deepseek_analyzer.analyze_paragraph(paragraph, risk_point, cot_result.get("reasoning", ""))
            
            # Add CoT reasoning to the result
            result["reasoning"] = cot_result.get("reasoning", "")
            result["paragraph"] = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            result["risk_point"] = risk_point
            result["model"] = "deepseek"
            
            results.append(result)
        
        return results
    
    def _run_minicpm_analysis(self, paragraphs: List[str], risk_point: str) -> List[Dict[str, Any]]:
        """Run analysis using MiniCPM model (fine-tuned small model)"""
        logger.info("Running analysis with MiniCPM model (fine-tuned small model)")
        results = []
        
        for i, paragraph in enumerate(paragraphs):
            if self.debug:
                logger.info(f"Analyzing paragraph {i+1}/{len(paragraphs)}")
            
            result = self.minicpm_analyzer.analyze_paragraph(paragraph, risk_point)
            result["paragraph"] = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            result["risk_point"] = risk_point
            result["model"] = "minicpm"
            
            results.append(result)
        
        return results
    
    def _run_cot_analysis(self, paragraphs: List[str], risk_point: str) -> List[Dict[str, Any]]:
        """Run analysis using Chain-of-Thought reasoning"""
        logger.info("Running analysis with Chain-of-Thought reasoning")
        results = []
        
        # Retrieve risk knowledge
        risk_knowledge = self.risk_retriever.get_risk_knowledge(risk_point)
        
        for i, paragraph in enumerate(paragraphs):
            if self.debug:
                logger.info(f"Analyzing paragraph {i+1}/{len(paragraphs)}")
            
            # Generate CoT prompt with risk knowledge
            cot_prompt = self.prompt_engineer.generate_cot_prompt(paragraph, risk_point, risk_knowledge)
            
            # Perform CoT reasoning
            result = self.cot_reasoner.perform_cot_analysis(paragraph, risk_point, cot_prompt)
            result["paragraph"] = paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            result["risk_point"] = risk_point
            result["model"] = "cot"
            
            results.append(result)
        
        return results
    
    def _run_collaboration_analysis(self, paragraphs: List[str], risk_point: str) -> List[Dict[str, Any]]:
        """
        Run analysis using model collaboration between DeepSeek (large model) and MiniCPM (fine-tuned small model)
        This implements the core research contribution of large-small model collaboration
        """
        logger.info("Running analysis with large-small model collaboration")
        
        # Process all paragraphs using batch analysis in the collaborator
        collaboration_results = self.model_collaborator.batch_analyze_paragraphs(paragraphs, risk_point)
        
        return collaboration_results
    
    def _run_all_models_analysis(self, paragraphs: List[str], risk_point: str) -> List[Dict[str, Any]]:
        """Run analysis using all models and combine results"""
        logger.info("Running analysis with all models")
        
        # Run each model separately
        deepseek_results = self._run_deepseek_analysis(paragraphs, risk_point)
        minicpm_results = self._run_minicpm_analysis(paragraphs, risk_point)
        cot_results = self._run_cot_analysis(paragraphs, risk_point)
        collaboration_results = self._run_collaboration_analysis(paragraphs, risk_point)
        
        # Save individual results
        all_models_results = {
            "deepseek": deepseek_results,
            "minicpm": minicpm_results,
            "cot": cot_results,
            "collaboration": collaboration_results
        }
        
        # Save combined results
        experiment_name = f"{self.experiment_id}_all_models_{risk_point.replace(' ', '_')}"
        results_path = os.path.join(self.output_dir, f"{experiment_name}.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_models_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved all models results to {results_path}")
        
        # Return collaboration results as the main results
        return collaboration_results
    
    def _save_combined_results(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """Save combined results for all risk points"""
        combined_results_path = os.path.join(self.output_dir, f"{self.experiment_id}_combined_results.json")
        
        try:
            with open(combined_results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved combined results to {combined_results_path}")
        except Exception as e:
            logger.error(f"Error saving combined results: {e}")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging level
    if args.debug:
        logger.setLevel("DEBUG")
    
    # Start timing
    start_time = time.time()
    
    # Run experiment
    experiment = RiskAnalysisExperiment(args)
    results = experiment.run_experiment()
    
    # Print execution time
    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    # Return success
    return 0 if results else 1

if __name__ == "__main__":
    exit(main())