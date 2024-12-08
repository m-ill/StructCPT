from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Optional, Tuple
import numpy as np

class RAGPipeline:
    def __init__(
        self,
        embedding_model_path: str,
        index_file: str,
        documents: List[str],
        llm_model_name: str = "skt/kogpt2-base-v2",
        device: Optional[str] = None
    ):
        """RAG 파이프라인 초기화 함수

        Args:
            embedding_model_path: 임베딩 모델 경로
            index_file: FAISS 인덱스 파일 경로
            documents: 문서 목록
            llm_model_name: LLM 모델 이름 (Hugging Face Hub)
            device: 사용할 디바이스 ('cuda', 'cpu', None)
        
        Raises:
            FileNotFoundError: 모델 파일이나 인덱스 파일이 없는 경우
            RuntimeError: 모델 로딩에 실패한 경우
        """
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        try:
            # 임베딩 모델 로드
            self.embedding_model = SentenceTransformer(embedding_model_path)
            self.embedding_model.to(self.device)
            
            # FAISS 인덱스 로드
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Index file not found: {index_file}")
            self.index = faiss.read_index(index_file)
            
            # 문서 저장
            self.documents = documents
            
            # LLM 모델과 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
            self.llm.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """쿼리에 대한 상위 k개 문서 검색 함수
        
        Args:
            query: 검색 쿼리
            top_k: 검색할 문서 개수
            
        Returns:
            튜플 (검색된 문서 리스트, 유사도 점수 리스트)
        """
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
            
            # FAISS 검색 수행
            distances, indices = self.index.search(
                query_embedding.cpu().numpy().reshape(1, -1),
                min(top_k, len(self.documents))
            )
            
            # 검색 결과 정리
            retrieved_documents = [self.documents[i] for i in indices[0]]
            similarity_scores = [float(score) for score in distances[0]]
            
            return retrieved_documents, similarity_scores
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            return [], []

    def generate(
        self,
        query: str,
        retrieved_documents: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        no_repeat_ngram_size: int = 2
    ) -> str:
        """LLM을 사용하여 답변 생성 함수
        
        Args:
            query: 사용자 질문
            retrieved_documents: 검색된 문서 목록
            max_length: 최대 생성 길이
            temperature: 생성 온도
            top_p: nucleus sampling 파라미터
            top_k: top-k sampling 파라미터
            no_repeat_ngram_size: 반복 방지를 위한 n-gram 크기
            
        Returns:
            생성된 답변
        """
        try:
            # 프롬프트 생성
            context = "\n".join(retrieved_documents)
            prompt = f"""다음 문서들을 참고하여, 아래 질문에 답변해주세요:

참고 문서:
{context}

질문: {query}

답변:"""

            # 토큰화
            inputs = self.tokenizer.encode_plus(
                prompt,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # 답변 생성
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # 디코딩 및 후처리
            generated_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_start = generated_answer.find("답변:") + len("답변:")
            final_answer = generated_answer[answer_start:].strip()
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."

    def run(
        self,
        query: str,
        top_k: int = 5,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> Tuple[str, List[str], List[float]]:
        """RAG 파이프라인 실행 함수
        
        Args:
            query: 사용자 질문
            top_k: 검색할 문서 개수
            max_length: 최대 생성 길이
            temperature: 생성 온도
            
        Returns:
            튜플 (생성된 답변, 검색된 문서 리스트, 유사도 점수 리스트)
        """
        try:
            # 문서 검색
            retrieved_documents, similarity_scores = self.retrieve(query, top_k)
            
            # 답변 생성
            answer = self.generate(
                query,
                retrieved_documents,
                max_length=max_length,
                temperature=temperature
            )
            
            return answer, retrieved_documents, similarity_scores
            
        except Exception as e:
            self.logger.error(f"Error during pipeline execution: {str(e)}")
            return "오류가 발생했습니다.", [], []

def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # 설정
        embedding_model_path = "./output/st_model"
        index_file = "output/saef.index"
        llm_model_name = "skt/kogpt2-base-v2"

        # 문서 로드
        documents = []
        raw_data_dir = "raw_data"
        
        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"Directory not found: {raw_data_dir}")
            
        for filename in os.listdir(raw_data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(raw_data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        # 문장 단위로 분리하고 빈 문장 제거
                        sentences = [s.strip() for s in text.split(".") if s.strip()]
                        documents.extend(sentences)
                except Exception as e:
                    logger.error(f"Error reading file {filepath}: {str(e)}")
                    continue

        # RAG 파이프라인 초기화
        rag_pipeline = RAGPipeline(
            embedding_model_path=embedding_model_path,
            index_file=index_file,
            documents=documents,
            llm_model_name=llm_model_name
        )

        # 대화 루프
        print("\nRAG 시스템이 준비되었습니다. '종료'를 입력하면 프로그램이 종료됩니다.\n")
        
        while True:
            query = input("\n질문을 입력하세요: ").strip()
            
            if query.lower() in ['종료', 'quit', 'exit']:
                break
                
            if not query:
                continue

            # RAG 파이프라인 실행
            answer, retrieved_docs, scores = rag_pipeline.run(query)
            
            # 결과 출력
            print("\n=== 답변 ===")
            print(answer)
            
            print("\n=== 참고한 문서 (상위 3개) ===")
            for doc, score in zip(retrieved_docs[:3], scores[:3]):
                print(f"\n- 문서 (유사도: {score:.3f}):")
                print(doc)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print("\n오류가 발생했습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    main()
