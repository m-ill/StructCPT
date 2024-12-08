import json
import os
import logging
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from rag_pipeline import RAGPipeline

@dataclass
class EvaluationExample:
    """평가 데이터 예시를 위한 데이터 클래스"""
    query: str
    relevant_docs: List[str]
    context: Optional[str] = None
    answer: Optional[str] = None

class RAGEvaluator:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        eval_data: List[Dict[str, Any]],
        top_k_values: List[int] = None
    ):
        """RAG 평가기 초기화
        
        Args:
            rag_pipeline: 평가할 RAG 파이프라인 객체
            eval_data: 평가 데이터 리스트
            top_k_values: 평가할 top-k 값들의 리스트
        """
        self.logger = logging.getLogger(__name__)
        self.rag_pipeline = rag_pipeline
        self.top_k_values = top_k_values or [1, 3, 5, 10]
        self.eval_examples = self._process_eval_data(eval_data)

    def _process_eval_data(self, eval_data: List[Dict[str, Any]]) -> List[EvaluationExample]:
        """평가 데이터 전처리 함수"""
        examples = []
        for item in eval_data:
            try:
                example = EvaluationExample(
                    query=item["query"],
                    relevant_docs=item["relevant_docs"],
                    context=item.get("context"),
                    answer=item.get("answer")
                )
                examples.append(example)
            except KeyError as e:
                self.logger.warning(f"필수 필드 누락: {e}. 해당 예시는 건너뜁니다.")
                continue
        return examples

    def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        top_k: int
    ) -> Dict[str, float]:
        """검색 성능 지표 계산 함수
        
        Args:
            retrieved_docs: 검색된 문서 리스트
            relevant_docs: 관련 문서 리스트
            top_k: 상위 k개 문서 수
            
        Returns:
            계산된 성능 지표 딕셔너리
        """
        retrieved_k_docs = retrieved_docs[:top_k]
        
        # Recall@k
        relevant_retrieved = set(retrieved_k_docs) & set(relevant_docs)
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
        
        # Precision@k
        precision = len(relevant_retrieved) / top_k
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            f"Recall@{top_k}": recall,
            f"Precision@{top_k}": precision,
            f"F1@{top_k}": f1
        }

    def calculate_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Mean Reciprocal Rank (MRR) 계산 함수"""
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate(self, batch_size: int = 32) -> Dict[str, float]:
        """RAG 모델 평가 실행 함수
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            평가 결과 딕셔너리
        """
        metrics = {
            "MRR": [],
        }
        for k in self.top_k_values:
            metrics.update({
                f"Recall@{k}": [],
                f"Precision@{k}": [],
                f"F1@{k}": []
            })

        # 배치 단위로 평가 수행
        for i in tqdm(range(0, len(self.eval_examples), batch_size), desc="Evaluating"):
            batch = self.eval_examples[i:i + batch_size]
            
            for example in batch:
                try:
                    # 문서 검색
                    retrieved_docs, _ = self.rag_pipeline.retrieve(
                        example.query,
                        top_k=max(self.top_k_values)
                    )
                    
                    # MRR 계산
                    mrr = self.calculate_mrr(retrieved_docs, example.relevant_docs)
                    metrics["MRR"].append(mrr)
                    
                    # 각 top-k 값에 대한 메트릭 계산
                    for k in self.top_k_values:
                        results = self.calculate_retrieval_metrics(
                            retrieved_docs,
                            example.relevant_docs,
                            k
                        )
                        for metric, value in results.items():
                            metrics[metric].append(value)
                            
                except Exception as e:
                    self.logger.error(f"평가 중 오류 발생: {str(e)}")
                    continue

        # 최종 결과 계산
        final_results = {}
        for metric, scores in metrics.items():
            if scores:
                final_results[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                }
            else:
                final_results[metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }

        return final_results

def load_evaluation_data(eval_data_file: str) -> List[Dict[str, Any]]:
    """평가 데이터 로드 함수
    
    Args:
        eval_data_file: 평가 데이터 파일 경로 (JSON)
        
    Returns:
        평가 데이터 리스트
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        json.JSONDecodeError: JSON 파싱 오류
    """
    if not os.path.exists(eval_data_file):
        raise FileNotFoundError(f"평가 데이터 파일을 찾을 수 없습니다: {eval_data_file}")
        
    try:
        with open(eval_data_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            
        if not isinstance(eval_data, list):
            raise ValueError("평가 데이터는 리스트 형태여야 합니다.")
            
        return eval_data
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON 파싱 오류: {str(e)}", e.doc, e.pos)

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
        eval_data_file = "eval_data.json"

        # 문서 로드
        documents = []
        raw_data_dir = "raw_data"
        
        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {raw_data_dir}")
            
        for filename in os.listdir(raw_data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(raw_data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        sentences = [s.strip() for s in text.split(".") if s.strip()]
                        documents.extend(sentences)
                except Exception as e:
                    logger.error(f"파일 읽기 오류 {filepath}: {str(e)}")
                    continue

        # RAG 파이프라인 초기화
        rag_pipeline = RAGPipeline(
            embedding_model_path=embedding_model_path,
            index_file=index_file,
            documents=documents,
            llm_model_name=llm_model_name
        )

        # 평가 데이터 로드
        eval_data = load_evaluation_data(eval_data_file)

        # 평가기 초기화
        evaluator = RAGEvaluator(rag_pipeline, eval_data)

        # 평가 실행
        results = evaluator.evaluate()

        # 결과 출력
        print("\n=== 평가 결과 ===")
        for metric, stats in results.items():
            print(f"\n{metric}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")

        # 결과 저장
        output_file = "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n평가 결과가 {output_file}에 저장되었습니다.")

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
