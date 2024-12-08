# StructCPT: 한국 건축 구조 분야 RAG 기반 검색 시스템

## 📌 개요

본 레포지토리는 논문 "*MAXIM 기반 대조 학습을 활용한 한국어 건축 구조 분야 지능형 정보 검색 시스템 개발*" (StructCPT: Contrastive Learning-based RAG System for Korean Architectural and Structural Engineering Information Retrieval)에서 제안된 StructCPT 모델을 구현하고 있습니다.

StructCPT는 한국 건축 구조 분야에 특화된 RAG(Retrieval-Augmented Generation) 기반 검색 시스템으로, SAEF(Structural AI Evaluation Framework) 데이터베이스와 MAXIM(Maximizing Semantic Relevance) 기반 대조 학습 기법을 활용하여, 쿼리와 문서 간의 의미적 연관성을 극대화하고 검색 정확도를 향상시킵니다.

## 💡 핵심 기술

- **SAEF 데이터베이스**: 한국 건축 구조 분야의 전문 지식을 체계적으로 수집하고 정제한 멀티도메인 데이터베이스
  - 전문 용어집
  - KDS 포함 설계 표준/코드
  - 교과서
  - 프로젝트 보고서 등

- **MAXIM 기반 대조 학습**: 쿼리와 문서 임베딩 간의 의미적 유사도를 극대화하는 도메인 특화 대조 학습 기법

- **듀얼 인코더(Dual-Encoder)**: 쿼리와 문서를 각각 고차원 임베딩 벡터로 변환하는 인코더

- **RAG 파이프라인**: 검색(Retrieval), 증강(Augmentation), 생성(Generation) 단계로 구성된 질의응답 시스템

## 📁 프로젝트 구조

```
structcpt/
├── data_preprocessing.py     # 데이터 수집, 정제, 가공
├── db_utils.py              # 데이터베이스 관련 유틸리티
├── train_embedding_model.py  # 임베딩 모델 학습 및 저장
├── rag_pipeline.py          # RAG 파이프라인 구현
├── evaluate_model.py        # 모델 평가
├── requirements.txt         # 의존성 패키지 목록
├── raw_data/               # 원본 데이터 (예시)
│   └── ...
├── output/                 # 학습된 모델, 임베딩, 평가 결과 저장
│   ├── st_model/          # Sentence Transformer 모델
│   └── saef.index         # FAISS 인덱스
├── eval_data.json         # 평가 데이터 (예시)
└── README.md              # 본 설명 문서
```

## 🚀 실행 방법

### 1. 환경 설정

- Python 3.8 이상
- 의존성 패키지 설치:
  ```bash
  pip install -r requirements.txt
  ```
- (Optional) CUDA 지원 GPU (학습 시간 단축)

### 2. 데이터 준비

1. `raw_data` 디렉토리에 SAEF 데이터베이스를 구성할 원본 데이터를 위치시킵니다.
   - 하위 디렉토리: glossary, standards, textbooks, reports
   - 지원 파일 형식: .txt, .pdf, .hwp, .docx

2. `data_preprocessing.py` 설정:
   - `load_stopwords` 함수: 불용어 목록 수정
   - `clean_text`, `tokenize`, `create_snippets` 함수 조정

3. `db_utils.py` 설정:
   - 데이터베이스 연결 정보 설정
   - 테이블 생성 쿼리 작성

4. 전처리 실행:
   ```bash
   python data_preprocessing.py
   ```

### 3. 임베딩 모델 학습

1. `train_embedding_model.py` 설정:
   - `model_name`: 사전 학습 언어 모델 선택
     - 기본값: bert-base-multilingual-cased
     - 추천 한국어 모델: snunlp/KR-Medium, klue/bert-base
   - 하이퍼파라미터 조정: epochs, learning_rate, temperature
   - MAXIM_ContrastiveLoss 손실함수 설정

2. 학습 실행:
   ```bash
   python train_embedding_model.py
   ```

### 4. RAG 파이프라인 실행

1. `rag_pipeline.py` 설정:
   - `llm_model_name`: LLM 모델 선택 (기본값: skt/kogpt2-base-v2)
   - `retrieve` 함수: top_k 설정
   - `generate` 함수: 답변 생성 설정

2. 파이프라인 실행:
   ```bash
   python rag_pipeline.py
   ```

### 5. 모델 평가

1. `eval_data.json` 작성:
   - query (질의)
   - relevant_docs (정답 문서 목록)

2. `evaluate_model.py` 실행:
   ```bash
   python evaluate_model.py
   ```

## 📝 참고

본 코드는 논문 "*MAXIM 기반 대조 학습을 활용한 한국어 건축 구조 분야 지능형 정보 검색 시스템 개발*"의 구현 예시입니다.

## 🤝 기여

버그 수정, 성능 개선, 기능 추가 등 어떠한 형태의 기여도 환영합니다.
Pull Request를 통해 기여해주시면 감사하겠습니다.

## 📜 라이선스

본 프로젝트는 MIT 라이선스를 따릅니다.

## 💬 문의

문의사항은 Issues 탭을 통해 남겨주세요.
