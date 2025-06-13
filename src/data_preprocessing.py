import os
import re
import random
import logging
from typing import List
import spacy

# 加载Spacy模型，用于简易的文本分类和分句
nlp = spacy.load("en_core_web_sm")

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def read_text_from_docx(file_path: str) -> str:
    """
    读取 docx 文件中的文本内容。
    请确保安装了 python-docx 包 (pip install python-docx)。
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请安装 python-docx 包")

    document = Document(file_path)
    text = []
    for para in document.paragraphs:
        text.append(para.text)
    return "\n".join(text)


def read_text_from_pdf(file_path: str) -> str:
    """
    读取 PDF 文件中的文本内容。
    请确保安装了 pdfminer.six 包 (pip install pdfminer.six)。
    """
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        raise ImportError("请安装 pdfminer.six 包")
    return extract_text(file_path)


def clean_text(text: str) -> str:
    """
    文本预处理：去除多余空白、特殊字符等。
    """
    text = re.sub(r'\s+', ' ', text)  # 合并多余空格
    text = re.sub(r'\n+', ' ', text)  # 去除多余换行
    return text.strip()


def is_title(text: str) -> bool:
    """
    判断一段文本是否为标题。
    基于常见的标题格式（如“第X条”或“甲方”）
    """
    return bool(re.match(r"^(第[一二三四五六七八九十]+条|[甲乙丙]方).*", text))


def segment_contract(text: str) -> List[str]:
    """
    将合同文本按中文标点断句，每 1~3 句合并为一段，
    同时保留结构化标号（如“一、”、“第二条”、“（二）”等）为新段开头。
    """
    # 先按中文句号、问号、感叹号断句
    sentences = re.split(r'(?<=[。！？])', text)

    # 合并结构性标题/编号为段落，如“一、”“第二条”“（一）”
    structural_pattern = re.compile(
        r'^\s*(第[一二三四五六七八九十百千]+条|[一二三四五六七八九十百千]+、|（[一二三四五六七八九十]）)')

    segments = []
    buffer = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # 如果是结构性段落编号，强制断段
        if structural_pattern.match(sent):
            if buffer:
                segments.append(''.join(buffer).strip())
                buffer = []
            segments.append(sent)
        else:
            buffer.append(sent)
            # 每 2-3 句组成一段
            if len(buffer) >= 2 and (len(buffer) >= 3 or random.random() < 0.5):
                segments.append(''.join(buffer).strip())
                buffer = []

    if buffer:
        segments.append(''.join(buffer).strip())

    return segments


def process_contract_file(file_path: str, output_dir: str) -> None:
    """
    处理单个合同文件，将预处理后的文本保存至 output_dir。
    根据文件后缀调用相应的读取函数（docx或pdf）。
    """
    logging.info(f"Processing file: {file_path}")
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in ['.docx']:
        text = read_text_from_docx(file_path)
    elif ext in ['.pdf']:
        text = read_text_from_pdf(file_path)
    else:
        logging.warning(f"Unsupported file type: {ext}. Skipping file: {file_path}")
        return

    paragraphs = segment_contract(text)

    # 保存预处理结果：每个文件生成一个 txt 文件，每行代表一个段落
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + ".txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for para in paragraphs:
            f.write(para + "\n")
    logging.info(f"Saved processed file to: {output_file}")


def process_all_contracts(input_dir: str, output_dir: str) -> None:
    """
    处理 input_dir 目录下所有合同文件，并保存预处理结果至 output_dir。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            process_contract_file(file_path, output_dir)


if __name__ == "__main__":
    input_dir = r"D:\Contract_star\Contract_risk_detection\data\raw"
    output_dir = r"D:\Contract_star\Contract_risk_detection\data\processed"
    process_all_contracts(input_dir, output_dir)


# 新增PDF解析功能
import pdfplumber

class ContractParser:
    def parse_pdf(self, file_path: str) -> List[str]:
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text())
        return texts

    # 增强预处理逻辑
    def preprocess(self, raw_text: List[str]) -> List[Dict]:
        processed = []
        for idx, text in enumerate(raw_text):
            if len(text) < 50: continue
            processed.append({
                'paragraph_id': f'p_{idx:04d}',
                'text': text.strip(),
                'metadata': {
                    'length': len(text),
                    'contains_legal_terms': any(term in text for term in ['甲方', '乙方', '违约责任'])
                }
            })
        return processed
