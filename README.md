# Bedrock 기반 챗봇 애플리케이션


## 실행 순서

1. 필요한 패키지 설치:
```
pip install -r requirements.txt
```

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


### 2. **Chat with Input**
```
streamlit run 2.chat-with-input.py
```
![Chat with Input](./images/2.chat-with-input.png)
- 1의 기능 포함
- 파일 입력을 "단기 보관 메모리(Short Term Memory)"로 활용
    - 지원하는 입력 유형 : 이미지, PDF, CSV, 파이썬 코드 등


### 3-1. **Chat RAG FAISS**
```
streamlit run 3-1.chat-rag-faiss.py
```
![Chat RAG FAISS](./images/3-1.chat-rag-faiss.png)
- 1의 기능 포함
- 파일 입력을 "장기 보관 메모리(Long Term Memory)"로 활용
    - 입력된 PDF 파일을 벡터로 변환 (Bedrock 임베딩 모델)
    - 변환된 벡터를 FAISS의 로컬 데이터베이스에 저장
    - 사용자 질문을 시맨틱 검색하여, 답변을 위한 컨텍스트로 활용
 
### 3-2. **Chat RAG OpenSearch Hybrid Retriever**
1. CloudFormation 파일(`cloudformation/setup_opensearch.yaml`)로 OpenSearch 클러스터 생성
    - 기존에 생성된 클러스터를 재사용 가능
3. `libs/opensearch.yml` 파일의 연결 정보 업데이트
4. 챗봇 애플리케이션 실행
```
streamlit run 3-1.chat-rag-faiss.py
```
![Chat with Input](./images/3-2.chat-rag-opensearch.png)
- 1의 기능 포함
- 파일 입력을 "장기 보관 메모리(Long Term Memory)"로 활용
    - 입력된 PDF 파일을 벡터로 변환 (Bedrock 임베딩 모델)
    - 변환된 벡터를 Amazon OpenSearch Service 클러스터에 저장 
    - 사용자 질문을 시맨틱 & 텍스트으로 검색하고, 검색 결과를 조합(앙상블)하여 답변을 위한 컨텍스트로 활용
