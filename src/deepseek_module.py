import json
import requests
from typing import List, Dict
from .utils import logger

class DeepseekAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['model_config']['deepseek']
        
        self.cot_template = """
        基于合同条款和风险知识库，请进行如下分析：
        1. 识别潜在风险段落（引用原文）
        2. 关联风险知识库中的风险类型
        3. 给出风险置信度评分（0-1）
        4. 详细推理过程（使用COT思维链）
        
        当前段落：{paragraph}
        风险知识库摘要：{knowledge_snippet}
        """

    def retrieve_knowledge(self, paragraph: str) -> str:
        # 简化的本地知识检索（后续可接入向量数据库）
        with open(self.config['risk_knowledge'], 'r') as f:
            knowledge = json.load(f)
        return '. '.join(knowledge['common_risks'][:3])

    def analyze_paragraph(self, paragraph: str) -> Dict:
        knowledge_snippet = self.retrieve_knowledge(paragraph)
        prompt = self.cot_template.format(
            paragraph=paragraph,
            knowledge_snippet=knowledge_snippet
        )

        response = requests.post(
            self.config['api_endpoint'],
            headers={"Authorization": f"Bearer {self.config['api_key']}"},
            json={
                "model": self.config['model_name'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config['temperature']
            }
        )

        return self._parse_response(response.json())

    def _parse_response(self, api_response: Dict) -> Dict:
        # 解析大模型返回的结构化数据
        try:
            content = api_response['choices'][0]['message']['content']
            return {
                'risk_paragraph': content.split('风险段落：')[-1].split('\n')[0].strip(),
                'risk_type': content.split('风险类型：')[-1].split('\n')[0].strip(),
                'confidence': float(content.split('置信度：')[-1].split('\n')[0].strip()),
                'reasoning': '\n'.join(content.split('推理过程：')[1:])
            }
        except Exception as e:
            logger.error(f"解析失败: {str(e)}")
            return {"error": str(e)}