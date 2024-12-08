import logging
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
import torch
import numpy as np
import faiss
from tqdm.auto import tqdm
from kss import split_sentences

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, document = self.data[idx]
        return InputExample(texts=[query, document])

class EmbeddingTrainer:
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        max_length: int = 256,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def initialize_model(self):
        try:
            # 모델 초기화
            word_embedding_model = models.Transformer(
                self.model_name,
                max_seq_length=self.max_length
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension()
            )
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"모델 초기화 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            raise

    def prepare_training_data(self, data_dir: str):
        try:
            train_data = []
            data_path = Path(data_dir)
            
            for file_path in data_path.glob("**/*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    # KSS로 문장 분리
                    sentences = split_sentences(text)
                    
                    for sentence in sentences:
                        train_data.append((sentence, sentence))
                        
            logger.info(f"데이터 준비 완료: {len(train_data)} 쌍의 문장")
            return train_data
        except Exception as e:
            logger.error(f"데이터 준비 실패: {e}")
            raise

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 3e-5,
        temperature: float = 0.05
    ):
        try:
            # MAXIM 손실 함수
            train_loss = losses.MultipleNegativesRankingLoss(
                model=self.model,
                scale=1.0/temperature
            )
            
            # 학습 실행
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=100,
                optimizer_params={'lr': learning_rate},
                output_path='output/st_model'
            )
            logger.info("모델 학습 완료")
        except Exception as e:
            logger.error(f"학습 실패: {e}")
            raise

    def generate_and_save_embeddings(
        self,
        documents: List[str],
        output_file: str = "output/saef.index",
        batch_size: int = 32
    ):
        try:
            # 임베딩 생성
            all_embeddings = []
            for i in tqdm(range(0, len(documents), batch_size)):
                batch = documents[i:i+batch_size]
                embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                all_embeddings.append(embeddings)

            # 전체 임베딩 합치기
            final_embeddings = np.vstack(all_embeddings)
            
            # FAISS 인덱스 생성 및 저장
            dimension = final_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(final_embeddings)
            index.add(final_embeddings)
            
            # 저장
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, output_file)
            logger.info(f"임베딩 저장 완료: {output_file}")
            
        except Exception as e:
            logger.error(f"임베딩 생성/저장 실패: {e}")
            raise

def main():
    try:
        # 트레이너 초기화
        trainer = EmbeddingTrainer()
        trainer.initialize_model()
        
        # 데이터 준비
        train_data = trainer.prepare_training_data("raw_data")
        train_dataset = EmbeddingDataset(train_data, trainer.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # 학습
        trainer.train(train_dataloader)
        
        # 임베딩 생성 및 저장
        documents = train_data  # 예시, 실제로는 전체 문서 데이터 사용
        trainer.generate_and_save_embeddings(documents)
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
