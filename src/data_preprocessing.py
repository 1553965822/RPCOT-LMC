import os
import docx
import re
import json
from typing import List, Dict, Any, Union
from src.utils import logger, ConfigManager, get_absolute_path, create_directory_if_not_exists

class ContractPreprocessor:
    """
    Module for reading and preprocessing contract files
    """
    
    def __init__(self):
        """Initialize the preprocessor with configuration"""
        self.config = ConfigManager()
        self.raw_dir = get_absolute_path(self.config.data_config.get('raw_dir', 'data/raw'))
        self.processed_dir = get_absolute_path(self.config.data_config.get('processed_dir', 'data/processed'))
        create_directory_if_not_exists(self.processed_dir)
    
    def read_docx(self, file_name: str) -> str:
        """
        Read a docx file and return its text content
        
        Args:
            file_name: Name of the docx file
            
        Returns:
            Text content of the file
        """
        # Construct the absolute file path
        file_path = os.path.join(self.raw_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        try:
            # Read the docx file
            doc = docx.Document(file_path)
            
            # Extract text from each paragraph
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
            logger.info(f"Successfully read docx file: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error reading docx file {file_path}: {e}")
            return ""
    
    def read_txt(self, file_name: str) -> str:
        """
        Read a txt file and return its content
        
        Args:
            file_name: Name of the txt file
            
        Returns:
            Text content of the file
        """
        # Construct the absolute file path
        file_path = os.path.join(self.raw_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Successfully read txt file: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error reading txt file {file_path}: {e}")
            return ""
    
    def read_contract_file(self, file_name: str) -> str:
        """
        Read a contract file based on its extension
        
        Args:
            file_name: Name of the contract file
            
        Returns:
            Text content of the file
        """
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension == '.docx':
            return self.read_docx(file_name)
        elif file_extension == '.txt':
            return self.read_txt(file_name)
        else:
            logger.error(f"Unsupported file extension: {file_extension}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the contract text
        
        Args:
            text: Raw contract text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Normalize common characters
        text = text.replace("，", ",").replace("。", ".")
        
        return text.strip()
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split contract text into paragraphs
        
        Args:
            text: Preprocessed contract text
            
        Returns:
            List of paragraphs
        """
        if not text:
            return []
        
        # Split by article numbers and line breaks
        paragraphs = []
        
        # First try to split by common article patterns (第x条, 第x章, etc.)
        article_pattern = re.compile(r'(第[一二三四五六七八九十百千万零\d]+[条章节][\s\S]*?)(?=第[一二三四五六七八九十百千万零\d]+[条章节]|$)')
        article_matches = list(article_pattern.finditer(text))
        
        if article_matches:
            for match in article_matches:
                article_text = match.group(1).strip()
                if article_text:
                    paragraphs.append(article_text)
        else:
            # If no article patterns found, split by numeric patterns (1. 2. etc.)
            numeric_pattern = re.compile(r'(\d+\.\s*[\s\S]*?)(?=\d+\.\s*|$)')
            numeric_matches = list(numeric_pattern.finditer(text))
            
            if numeric_matches:
                for match in numeric_matches:
                    paragraph_text = match.group(1).strip()
                    if paragraph_text:
                        paragraphs.append(paragraph_text)
            else:
                # If no patterns found, split by line breaks
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Filter out very short paragraphs (likely headers or noise)
        return [p for p in paragraphs if len(p) > 10]
    
    def process_contract(self, file_name: str) -> Dict[str, Any]:
        """
        Process a contract file and prepare it for analysis
        
        Args:
            file_name: Name of the contract file
            
        Returns:
            Dictionary containing preprocessed data
        """
        # Read the contract file
        raw_text = self.read_contract_file(file_name)
        
        if not raw_text:
            logger.error(f"Failed to read contract file: {file_name}")
            return {"file_name": file_name, "paragraphs": [], "success": False}
        
        # Preprocess the text
        processed_text = self.preprocess_text(raw_text)
        
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(processed_text)
        
        # Save processed text to file
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(self.processed_dir, f"{base_name}.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(paragraphs))
            
            logger.info(f"Successfully processed contract file: {file_name}")
            logger.info(f"Extracted {len(paragraphs)} paragraphs")
            
            return {
                "file_name": file_name,
                "paragraphs": paragraphs,
                "paragraph_count": len(paragraphs),
                "output_file": output_file,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error saving processed file {output_file}: {e}")
            return {
                "file_name": file_name,
                "paragraphs": paragraphs,
                "success": False
            }
    
    def get_paragraphs(self, file_name: str) -> List[str]:
        """
        Get paragraphs from a processed contract file
        
        Args:
            file_name: Name of the contract file
            
        Returns:
            List of paragraphs
        """
        base_name = os.path.splitext(file_name)[0]
        processed_file = os.path.join(self.processed_dir, f"{base_name}.txt")
        
        # Check if processed file exists
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                logger.info(f"Loaded {len(paragraphs)} paragraphs from processed file")
                return paragraphs
            except Exception as e:
                logger.error(f"Error loading processed file {processed_file}: {e}")
        
        # If processed file doesn't exist or loading fails, process the file
        result = self.process_contract(file_name)
        return result.get("paragraphs", [])