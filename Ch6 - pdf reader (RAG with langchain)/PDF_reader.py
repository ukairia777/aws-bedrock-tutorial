##### 기본 정보 입력 #####
# Streamlit 패키지 추가
import streamlit as st
# PDF reader
from PyPDF2 import PdfReader
# Langchain 패키지들
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
import boto3
import json

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

##### 메인 함수 #####
def main():
    st.title("🤖 PDF analyzer📜")
    st.image('ai.png', width=200)
    # 메인공간
    st.markdown('---')
    st.subheader("Please upload a PDF file and ask your question.")
    # PDF 파일 받기
    pdf = st.file_uploader(" ", type="pdf")
    if pdf is not None:
        # PDF 파일 텍스트 추출하기
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # 청크 단위로 분할하기
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        st.markdown('---')
        st.subheader("질문을 입력하세요")
        # 사용자 질문 받기
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # 임베딩/ 시멘틱 인덱스

            embeddings = BedrockEmbeddings(
                region_name='us-east-1',
                endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            ) 
            # embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API"])
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            docs = knowledge_base.similarity_search(user_question)

            # 질문하기
            llm = BedrockChat(model_kwargs={"temperature": 0},
                        model_id="anthropic.claude-v2",
                        client=bedrock
                    )

            # 프롬프트
            prompt_template = '''
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. You must answer in Korean.

            {context}

            Question: {question}
            Helpful Answer:"""
            '''

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
            response = chain.run(input_documents=docs, question=user_question)
            # 답변결과
            st.info(response)

if __name__=='__main__':
    main()