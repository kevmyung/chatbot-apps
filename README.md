# Bedrock 기반 챗봇 애플리케이션


## 실행 순서

1. 필요한 패키지 설치:
```
pip install -r requirements.txt
```
*여러 라이브러리가 설치되기 때문에, 사용 환경에 따라 의존성 오류가 발생할 수 있습니다. 설치 전에, 기존 환경과의 버전 호환성을 체크해주세요.

2. 원하는 챗봇 애플리케이션 실행:
```
streamlit run {Chatbot file}.py
```

## 예시

### 1. **Basic Chat**
```
streamlit run 1.basic-chat.py
```
![Basic Chat](./images/1.basic-chat.png)
- Bedrock 기반 모델 선택
- 시스템 프롬프트 제공
- 대화용 메모리 버퍼
- Base 코드 활용 : [Bedrock-ChatBot-with-LangChain-and-Streamlit](https://github.com/davidshtian/Bedrock-ChatBot-with-LangChain-and-Streamlit)

### 2. **Chat with Input**
```
streamlit run 2.chat-with-input.py
```
![Chat with Input](./images/2.chat-with-input.png)
- 기본 챗봇 기능 (`1. Basic Chat`)
- 파일 입력을 "단기 보관 메모리(Short Term Memory)"로 활용
    - 지원하는 입력 유형 : 이미지, PDF, CSV, 파이썬 코드 등


### 3-1. **Chat RAG FAISS with Image Context**
```
streamlit run 3-1.chat-rag-faiss.py
```
![Chat RAG FAISS](./images/3-1.chat-rag-faiss.png)
- 기본 챗봇 기능 (`1. Basic Chat`)
- 파일 입력을 "장기 보관 메모리(Long Term Memory)"로 활용
    - 입력된 PDF 파일을 벡터로 변환 (Bedrock 임베딩 모델)
    - 변환된 벡터를 FAISS의 로컬 데이터베이스에 저장
    - 사용자 질문을 시맨틱 검색하여, 답변을 위한 컨텍스트로 활용
- PDF 페이지를 이미지로 저장한 후, 검색 결과의 컨텍스트로 제공
    - 추가 라이브러리 `sudo apt-get install poppler-utils`
 
### 3-2. **Chat RAG OpenSearch with Hybrid Retriever**
1. CloudFormation 파일(`cloudformation/setup_opensearch.yaml`)로 OpenSearch 클러스터 생성
    - 기존에 생성된 클러스터를 재사용 가능
3. `libs/opensearch.yml` 파일의 연결 정보 업데이트
4. 챗봇 애플리케이션 실행
```
streamlit run 3-2.chat-rag-opensearch-hybrid.py
```
![Chat with Input](./images/3-2.chat-rag-opensearch.png)
- 기본 챗봇 기능 (`1. Basic Chat`)
- 파일 입력을 "장기 보관 메모리(Long Term Memory)"로 활용
    - 입력된 PDF 파일을 벡터로 변환 (Bedrock 임베딩 모델)
    - 변환된 벡터를 Amazon OpenSearch Service 클러스터에 저장 
    - 사용자 질문을 시맨틱 & 텍스트으로 검색하고, 검색 결과를 조합(앙상블)하여 답변을 위한 컨텍스트로 활용
- OpenSearch Hybrid Search 코드 활용 : [aws-ai-ml-workshop-kr](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/utils/rag.py)

### 4. **Chat SQL Agent**
```
streamlit run 4.chat-sql-agent.py
```
![Chat SQL Agent](./images/4.chat-sql-agent.png)
- 사용자의 자연어 질문을 Agent 기반으로 SQL 쿼리 변환/실행
- 샘플 데이터베이스([Chinook DB](https://github.com/lerocha/chinook-database))를 활용하거나, 데이터베이스 URI 입력
- Langchain 라이브러리를 활용해 쿼리 변환 및 DB 조회 
- DB 스키마(테이블/컬럼)에 대한 상세 Description을 참고하기 위한 custom agent 로직 추가
- **자동 시각화 기능 추가 구현 필요**
