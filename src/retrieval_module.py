import json
import os
from typing import List, Dict, Any, Tuple, Optional

from src.utils import logger, ConfigManager, get_absolute_path

class RiskKnowledgeRetriever:
    """
    Module for loading risk knowledge base and retrieving relevant paragraphs
    """
    
    def __init__(self):
        """Initialize the risk knowledge retriever with configuration"""
        self.config = ConfigManager()
        self.risk_knowledge_file = get_absolute_path(self.config.data_config.get('risk_knowledge_file', 'data/risk_knowledge.json'))
        self.retrieval_threshold = float(self.config.experiment_config.get('retrieval_threshold', 0.6))
        self.risk_knowledge = self._load_risk_knowledge()
    
    def _load_risk_knowledge(self) -> Dict[str, Any]:
        """
        Load risk knowledge from JSON file
        
        Returns:
            Risk knowledge dictionary
        """
        try:
            if not os.path.exists(self.risk_knowledge_file):
                logger.warning(f"Risk knowledge file not found: {self.risk_knowledge_file}")
                return {"risks": {}}
            
            with open(self.risk_knowledge_file, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)
            
            logger.info(f"Successfully loaded risk knowledge from {self.risk_knowledge_file}")
            logger.info(f"Loaded {len(knowledge.get('risks', {}))} risk categories")
            
            return knowledge
        except Exception as e:
            logger.error(f"Error loading risk knowledge: {e}")
            return {"risks": {}}
    
    def get_risk_points(self) -> List[str]:
        """
        Get list of available risk points
        
        Returns:
            List of risk point names
        """
        return list(self.risk_knowledge.get("risks", {}).keys())
    
    def get_risk_description(self, risk_point: str) -> str:
        """
        Get description for a specific risk point
        
        Args:
            risk_point: Name of the risk point
            
        Returns:
            Description of the risk point
        """
        risk_info = self.risk_knowledge.get("risks", {}).get(risk_point, {})
        return risk_info.get("description", f"{risk_point}风险点")
    
    def get_risk_examples(self, risk_point: str) -> List[str]:
        """
        Get examples for a specific risk point
        
        Args:
            risk_point: Name of the risk point
            
        Returns:
            List of example texts
        """
        risk_info = self.risk_knowledge.get("risks", {}).get(risk_point, {})
        return risk_info.get("examples", [])
    
    def retrieve_relevant_risk_points(self, paragraph: str) -> List[Tuple[str, float]]:
        """
        Retrieve relevant risk points for a given paragraph using simple keyword matching
        Note: In a production system, this would use advanced semantic search techniques
        
        Args:
            paragraph: Contract paragraph text
            
        Returns:
            List of (risk_point, relevance_score) tuples sorted by relevance
        """
        relevance_scores = []
        
        for risk_point, risk_info in self.risk_knowledge.get("risks", {}).items():
            # Simple keyword matching (for demonstration purposes)
            score = 0.0
            
            # Check if risk name appears in paragraph
            if risk_point in paragraph:
                score += 0.5
            
            # Check if description keywords appear in paragraph
            description = risk_info.get("description", "")
            for keyword in description.split():
                if len(keyword) > 1 and keyword in paragraph:
                    score += 0.3
            
            # Check if examples have similar content
            for example in risk_info.get("examples", []):
                for keyword in example.split():
                    if len(keyword) > 1 and keyword in paragraph:
                        score += 0.2
            
            # Normalize score
            score = min(score, 1.0)
            
            # Add to results if above threshold
            if score >= self.retrieval_threshold:
                relevance_scores.append((risk_point, score))
        
        # Sort by relevance score (descending)
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return relevance_scores
    
    def generate_knowledge_context(self, risk_point: str) -> str:
        """
        Generate context information for a specific risk point to enhance prompts
        
        Args:
            risk_point: Name of the risk point
            
        Returns:
            Formatted context string
        """
        description = self.get_risk_description(risk_point)
        examples = self.get_risk_examples(risk_point)
        
        context = f"风险点：{risk_point}\n"
        context += f"描述：{description}\n"
        
        if examples:
            context += "示例：\n"
            for i, example in enumerate(examples, 1):
                context += f"{i}. {example}\n"
        
        return context