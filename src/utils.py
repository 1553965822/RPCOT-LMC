import os
import sys
import logging
import yaml
import json
import re
from typing import Dict, Any, List, Optional, Union, Callable

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('rpcot_lmc')

def get_absolute_path(relative_path: str) -> str:
    """
    Convert a relative path to absolute path from project root
    
    Args:
        relative_path: Path relative to project root
        
    Returns:
        Absolute path
    """
    # Get the project root directory
    if os.path.isabs(relative_path):
        return relative_path
        
    # Determine project root (parent directory of the src directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    project_root = os.path.dirname(current_dir)  # parent directory of src
    
    # Create absolute path
    return os.path.join(project_root, relative_path)

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

class ConfigManager:
    """
    Class to manage configuration from YAML file
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config_file = get_absolute_path('config/config.yaml')
        self.config = self._load_config()
        
        # Parse individual config sections
        self.data_config = self.config.get('data_config', {})
        self.model_config = self.config.get('model_config', {})
        self.experiment_config = self.config.get('experiment_config', {})
        
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {self.config_file}")
            return self._create_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            return {}
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration if config file doesn't exist
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            'data_config': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'risk_knowledge_file': 'data/risk_knowledge.json'
            },
            'model_config': {
                'deepseek_model': 'deepseek-llm',
                'deepseek_temperature': 0.3,
                'deepseek_max_tokens': 350,
                'minicpm_model': 'minicpm',
                'minicpm_temperature': 0.2,
                'minicpm_max_tokens': 400,
                'sentence_transformer_model': 'all-MiniLM-L6-v2'
            },
            'experiment_config': {
                'retrieval_threshold': 0.6,
                'collaboration_threshold': 0.2,
                'cot_max_tokens': 300,
                'prompt_max_tokens': 150,
                'results_dir': 'experiments/results',
                'evaluation_positive_label': '包含'
            }
        }
        
        # Create config directory if it doesn't exist
        config_dir = os.path.dirname(self.config_file)
        create_directory_if_not_exists(config_dir)
        
        # Save default config
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Created default configuration at {self.config_file}")
        return default_config
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation path
        
        Args:
            key_path: Dot notation path (e.g., 'data_config.raw_dir')
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_value(self, key_path: str, value: Any) -> bool:
        """
        Set a configuration value using dot notation path
        
        Args:
            key_path: Dot notation path (e.g., 'data_config.raw_dir')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        # Update the specific config section
        if keys[0] == 'data_config':
            self.data_config = self.config.get('data_config', {})
        elif keys[0] == 'model_config':
            self.model_config = self.config.get('model_config', {})
        elif keys[0] == 'experiment_config':
            self.experiment_config = self.config.get('experiment_config', {})
        
        # Save to file
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Updated configuration at {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

def call_llm_api(prompt: str, max_tokens: int = 200, temperature: float = 0.3, model: str = None) -> str:
    """
    Call LLM API with given parameters.
    This is a placeholder function that simulates LLM API call for development.
    In a real implementation, this would call an actual API.
    
    Args:
        prompt: Prompt text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        model: Model name
    
    Returns:
        Generated text response
    """
    # Placeholder implementation
    logger.info(f"LLM API called with model={model}, temp={temperature}, max_tokens={max_tokens}")
    logger.debug(f"Prompt: {prompt[:100]}...")
    
    # In a real implementation, this would call OpenAI, DeepSeek, etc.
    # For development, return a simulated response
    if "风险" in prompt:
        decision = "包含" if "违约" in prompt else "不包含"
        confidence = 0.85 if "违约" in prompt else 0.65
        
        return json.dumps({
            "decision": decision,
            "confidence": confidence,
            "explanation": f"该条款{'存在违约风险' if decision == '包含' else '不存在明显风险'}，因为...",
            "risk_factors": ["条款不明确", "责任划分不清"] if decision == "包含" else [],
            "suggestions": ["建议明确违约责任", "建议增加保障条款"] if decision == "包含" else []
        }, ensure_ascii=False)
    
    return "模拟LLM响应。这是一个占位符，需要替换为实际API调用。"

def normalize_decision(text: str) -> str:
    """
    Normalize decision text to standard format
    
    Args:
        text: Raw decision text
        
    Returns:
        Normalized decision: "包含" or "不包含"
    """
    text_lower = text.lower()
    
    if any(pos in text_lower for pos in ["包含", "存在", "有风险", "发现风险"]):
        return "包含"
    elif any(neg in text_lower for neg in ["不包含", "没有", "无风险", "不存在风险"]):
        return "不包含"
    else:
        # No clear decision found
        return ""