from langchain.embeddings import BedrockEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.chat_models import BedrockChat
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import boto3
import gradio as gr
import random
import time

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

loader = PyPDFLoader("./2020_경제금융용어 700선_게시.pdf")
texts = loader.load_and_split()

embedding = BedrockEmbeddings(
        region_name='us-east-1',
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        model_id="amazon.titan-embed-text-v1"
    ) 

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding)

# 벡터DB의 개수 확인
print('벡터 DB 내장된 벡터의 개수 :', vectordb._collection.count())

# 유사도가 높은 문서 2개만 추출. k = 2
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=BedrockChat(model_kwargs={"temperature": 0},
                        model_id="anthropic.claude-v2",
                        client=bedrock
                    ),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

def get_chatbot_response(chatbot_response):
    return chatbot_response['result'].strip()

# 인터페이스를 생성.
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="경제금융용어 챗봇") # 경제금융용어 챗봇 레이블을 좌측 상단에 구성
    msg = gr.Textbox(label="질문해주세요!")  # 하단의 채팅창의 레이블
    clear = gr.Button("대화 초기화")  # 대화 초기화 버튼

    # 챗봇의 답변을 처리하는 함수
    def respond(message, chat_history):
      result = qa_chain(message)
      bot_message = result['result']

      # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가.
      chat_history.append((message, bot_message))
      return "", chat_history

    # 사용자의 입력을 제출(submit)하면 respond 함수가 호출.
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # '초기화' 버튼을 클릭하면 채팅 기록을 초기화.
    clear.click(lambda: None, None, chatbot, queue=False)

# 인터페이스 실행.
demo.launch(debug=True)