StructCPT: 한국 건축 구조 분야 RAG 기반 검색 시스템
개요
본 레포지토리는 논문 "MAXIM 기반 대조 학습을 활용한 한국어 건축 구조 분야 지능형 정보 검색 시스템 개발 (StructCPT: Contrastive Learning-based RAG System for Korean Architectural and Structural Engineering Information Retrieval)"에서 제안된 StructCPT 모델을 구현하고 있습니다. StructCPT는 한국 건축 구조 분야에 특화된 RAG(Retrieval-Augmented Generation) 기반 검색 시스템으로, SAEF(Structural AI Evaluation Framework) 데이터베이스와 MAXIM(Maximizing Semantic Relevance) 기반 대조 학습 기법을 활용하여, 쿼리와 문서 간의 의미적 연관성을 극대화하고 검색 정확도를 향상시킵니다.

핵심 기술
SAEF 데이터베이스: 한국 건축 구조 분야의 전문 지식을 체계적으로 수집하고 정제한 멀티도메인 데이터베이스 (전문 용어집, KDS 포함 설계 표준/코드, 교과서, 프로젝트 보고서 등)

MAXIM 기반 대조 학습: 쿼리와 문서 임베딩 간의 의미적 유사도를 극대화하는 도메인 특화 대조 학습 기법

듀얼 인코더(Dual-Encoder): 쿼리와 문서를 각각 고차원 임베딩 벡터로 변환하는 인코더

RAG(Retrieval-Augmented Generation) 파이프라인: 검색(Retrieval), 증강(Augmentation), 생성(Generation) 단계로 구성된 질의응답 시스템

프로젝트 구조
structcpt/
├── data_preprocessing.py     # 데이터 수집, 정제, 가공
├── db_utils.py               # 데이터베이스 관련 유틸리티
├── train_embedding_model.py  # 임베딩 모델 학습 및 저장
├── rag_pipeline.py           # RAG 파이프라인 구현
├── evaluate_model.py         # 모델 평가
├── requirements.txt          # 의존성 패키지 목록
├── raw_data/                 # 원본 데이터 (예시)
│   └── ...
├── output/                   # 학습된 모델, 임베딩, 평가 결과 저장
│   ├── st_model/             # Sentence Transformer 모델
│   └── saef.index            # FAISS 인덱스
├── eval_data.json            # 평가 데이터 (예시)
└── README.md                 # 본 설명 문서
Use code with caution.
실행 방법
1. 환경 설정
Python 3.8 이상

의존성 패키지 설치: pip install -r requirements.txt

(Optional) CUDA 지원 GPU (학습 시간 단축)

2. 데이터 준비
raw_data 디렉토리에 SAEF 데이터베이스를 구성할 원본 데이터를 위치시킵니다.

하위 디렉토리에 glossary, standards, textbooks, reports 와 같이 데이터를 분류하여 저장하는 것을 권장합니다.

파일 형식은 .txt, .pdf, .hwp, .docx 등을 지원합니다.

data_preprocessing.py를 수정하여 데이터 수집, 정제, 가공 로직을 필요에 맞게 조정합니다.

load_stopwords 함수에서 불용어 목록(stopwords.txt)을 필요에 따라 수정합니다.

clean_text, tokenize, create_snippets 함수에서 정제, 토크나이징, 스니펫 생성 방식을 변경할 수 있습니다.

db_utils.py를 수정하여 데이터베이스 연결 정보를 설정합니다.

connect_to_database 함수에서 사용하는 데이터베이스 종류, 호스트, 사용자, 비밀번호 등을 환경에 맞게 수정합니다.

create_tables 함수에서 필요한 테이블 생성 쿼리를 작성합니다.

data_preprocessing.py를 실행하여 데이터를 전처리하고 데이터베이스에 저장합니다.

python data_preprocessing.py
Use code with caution.
Bash
3. 임베딩 모델 학습
train_embedding_model.py를 수정하여 학습 파라미터를 조정합니다.

model_name 변수에서 사용할 사전 학습 언어 모델(PLM)을 지정합니다. (기본값: bert-base-multilingual-cased, 한국어 모델 추천: snunlp/KR-Medium, klue/bert-base 등)

epochs, learning_rate, temperature 등의 하이퍼파라미터를 조정할 수 있습니다.

train_model 함수 내 MAXIM 대조 학습에 사용되는 MAXIM_ContrastiveLoss 손실함수를 변경할 수 있습니다.

train_embedding_model.py를 실행하여 임베딩 모델을 학습하고, 학습된 모델과 임베딩을 저장합니다.

python train_embedding_model.py
Use code with caution.
Bash
학습된 Sentence Transformer 모델은 output/st_model 디렉토리에 저장됩니다.

문서 임베딩은 output/saef.index 파일에 FAISS 인덱스 형태로 저장됩니다.

4. RAG 파이프라인 실행
rag_pipeline.py를 수정하여 RAG 파이프라인을 조정합니다.

llm_model_name 변수에서 답변 생성에 사용할 LLM을 지정합니다. (기본값: skt/kogpt2-base-v2)

retrieve 함수에서 검색할 상위 문서 개수(top_k)를 변경할 수 있습니다.

generate 함수에서 LLM의 답변 생성 설정을 조정할 수 있습니다. (max_length, num_return_sequences, no_repeat_ngram_size, top_k, top_p, temperature 등)

rag_pipeline.py를 실행하여 RAG 파이프라인을 테스트합니다.

python rag_pipeline.py
Use code with caution.
Bash
사용자 쿼리를 입력하면, RAG 파이프라인이 답변을 생성하여 출력합니다.

5. 모델 평가
eval_data.json 파일에 평가 데이터를 JSON 형식으로 작성합니다.

각 데이터는 query (질의)와 relevant_docs (정답 문서 목록)로 구성됩니다.

evaluate_model.py를 수정하여 평가 설정을 조정합니다.

top_k_values 변수에서 Recall@k, Precision@k를 계산할 때 사용할 k 값을 지정합니다.

evaluate_model.py를 실행하여 모델을 평가합니다.

python evaluate_model.py
Use code with caution.
Bash
평가 결과(Recall@k, MRR, Precision@k)가 출력됩니다.

참고
본 코드는 논문 "MAXIM 기반 대조 학습을 활용한 한국어 건축 구조 분야 지능형 정보 검색 시스템 개발"의 구현 예시입니다.

코드를 실행하기 전에 raw_data 디렉토리에 데이터를 준비하고, data_preprocessing.py, db_utils.py, train_embedding_model.py, rag_pipeline.py, evaluate_model.py 파일의 설정을 확인해야 합니다.

필요에 따라 코드를 수정하여 성능을 개선하거나 기능을 추가할 수 있습니다.

기여
버그 수정, 성능 개선, 기능 추가 등 어떠한 형태의 기여도 환영합니다.

Pull Request를 통해 기여해주시면 감사하겠습니다.

라이선스
본 프로젝트는 MIT 라이선스를 따릅니다.

문의
문의사항은 Issues 탭을 통해 남겨주세요.
