import os
import re
from pathlib import Path
from typing import List, Set, Generator
from dataclasses import dataclass
from konlpy.tag import Okt
import textract
import PyPDF2
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import yaml

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    max_tokens: int = 200
    batch_size: int = 1000
    supported_extensions: Set[str] = frozenset({'.txt', '.pdf', '.docx'})

class TextProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.okt = Okt()  # 형태소 분석기 한 번만 초기화
        
    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return ' '.join(
                    page.extract_text() 
                    for page in pdf_reader.pages
                )
        except Exception as e:
            logger.error(f"PDF 처리 오류 {pdf_path}: {e}")
            return ""

    def process_file(self, file_path: str) -> Generator[str, None, None]:
        if not Path(file_path).suffix.lower() in ProcessingConfig.supported_extensions:
            logger.warning(f"지원하지 않는 파일 형식: {file_path}")
            return

        try:
            text = (
                self.extract_text_from_pdf(file_path)
                if file_path.lower().endswith('.pdf')
                else textract.process(file_path).decode('utf-8')
            )
            
            cleaned_text = self.clean_text(text)
            yield from self.create_snippets(cleaned_text)
            
        except Exception as e:
            logger.error(f"파일 처리 오류 {file_path}: {e}")

    def clean_text(self, text: str) -> str:
        # 더 정교한 텍스트 정제 규칙 적용
        text = re.sub(r'[^가-힣A-Za-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_snippets(self, text: str) -> Generator[str, None, None]:
        current_snippet = []
        current_length = 0
        
        for sentence in text.split('. '):
            tokens = self.okt.morphs(sentence)
            
            if current_length + len(tokens) <= ProcessingConfig.max_tokens:
                current_snippet.extend(tokens)
                current_length += len(tokens)
            else:
                if current_snippet:
                    yield ' '.join(current_snippet)
                current_snippet = tokens
                current_length = len(tokens)

        if current_snippet:
            yield ' '.join(current_snippet)

def main():
    processor = TextProcessor('config.yaml')
    data_dir = Path("raw_data")

    with ThreadPoolExecutor() as executor:
        files = list(data_dir.glob('**/*'))  # 하위 디렉토리 포함 모든 파일
        
        for file in tqdm(files, desc="Processing files"):
            if file.is_file():
                for snippet in processor.process_file(str(file)):
                    # 여기서 데이터베이스에 저장하거나 다른 처리를 수행
                    logger.info(f"Processed snippet: {snippet[:100]}...")

if __name__ == "__main__":
    main()
