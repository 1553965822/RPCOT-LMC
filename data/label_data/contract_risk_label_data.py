import os
import mammoth
import docx
import pdfplumber
import random
import json
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# 中文分句（替代英文分句器）
def chinese_sent_tokenize(text):
    return re.split(r'(?<=[。！？])', text)

# 中文风险点词典
risk_points = {
    "权利义务失衡": ["无需", "仅需"],
    "违约责任不明确": ["不得", "原因"],
    "加重对方责任": ["违约", "支付"],
    "单方解除权": ["单方面", "无条件"],
    "责任规避": ["自行承担","不承担"],
    "争议解决机制缺失": ["争议", "管辖"]
}

# .doc 转 .docx（原文件保留）
def convert_doc_to_docx(doc_path):
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    docx_path = doc_path + "x"
    try:
        doc = word.Documents.Open(doc_path)
        doc.SaveAs(docx_path, 12)
        doc.Close()
    except Exception as e:
        print(f"转换失败：{doc_path} -> {e}")
        docx_path = None
    finally:
        word.Quit()
    return docx_path

# 读取docx文件
def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

# 读取pdf文件
def extract_pdf_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 核心段落筛选 + 风险点比对 + 标签标注
def extract_risk_sentences(text, risk_points, max_per_doc=5):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 10]
    selected_data = []

    for para in paragraphs:
        # 修改第59行附近的匹配逻辑
        # 原代码
        # if any(keyword in para...):
        # 建议改为
        if any(
            re.search(rf'\b{keyword}\b', para) 
            for risk in risk_points.values() 
            for keyword in risk
        ):
            sentence_candidates = chinese_sent_tokenize(para)
            sentence_candidates = [s.strip() for s in sentence_candidates if len(s.strip()) > 5]
            if not sentence_candidates:
                continue
            sentence = random.choice(sentence_candidates)
            matched = False
            for risk, keywords in risk_points.items():
                if any(kw in sentence for kw in keywords):
                    selected_data.append({
                        "paragraph": sentence,
                        "risk_point": risk,
                        "label": "包含"
                    })
                    matched = True
                    break
            if not matched:
                selected_data.append({
                    "paragraph": sentence,
                    "risk_point": random.choice(list(risk_points.keys())),
                    "label": "不包含"
                })

        # 控制每份合同最多抽样一定数量
        if len(selected_data) >= max_per_doc:
            break

    return selected_data

def process_contracts(source_dir, output_path):
    result = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()
            text = ""

            # 如果是.doc，转为.docx
            if file_lower.endswith(".doc") and not file_lower.endswith(".docx"):
                print(f"转换 .doc 为 .docx：{file}")
                docx_path = convert_doc_to_docx(file_path)
                if not docx_path:
                    continue
                text = extract_docx_text(docx_path)

            elif file_lower.endswith(".docx"):
                text = extract_docx_text(file_path)

            elif file_lower.endswith(".pdf"):
                text = extract_pdf_text(file_path)

            else:
                continue

            if not text.strip():
                continue

            contract_key = file.replace(".docx", ".txt").replace(".doc", ".txt").replace(".pdf", ".txt")
            result[contract_key] = extract_risk_sentences(text, risk_points)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "合同风险段落标注数据.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 合同风险数据已保存到：{output_file}")

# 参数配置
if __name__ == "__main__":
    source_folder = r"D:\Contract_star\Contract_risk_detection\data\raw"
    output_folder = r"D:\Contract_star\Contract_risk_detection\data\label_data"
    process_contracts(source_folder, output_folder)
