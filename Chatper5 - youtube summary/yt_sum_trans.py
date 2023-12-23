##### 기본 정보 불러오기 ####

# Streamlit 패키지 추가
import streamlit as st
# 정규표현식 검색
import re
# Langchain 패키지들
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import BedrockChat
import boto3
import json

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

def trans(text):
    body = json.dumps({
        "prompt": """\n\nHuman: 당신은 영한 번역가이자 요약가입니다. 들어오는 모든 입력을 한국어로 번역하고 불렛 포인트 요약을 사용하여 답변하시오. 반드시 불렛 포인트 요약이어야만 합니다.
        질문: """ + text + """
        \n\nAssistant:""",
        "max_tokens_to_sample": 1000,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    return response_body.get('completion')

def youtube_url_check(url):
    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$'
    match = re.match(pattern, url)
    return match is not None

##### 메인 함수 #####
def main():
    st.set_page_config(page_title="YouTube Summerize", layout="wide")
    # session state 초기화
    if "flag" not in st.session_state:
        st.session_state["flag"] = True
    if "summerize" not in st.session_state:
        st.session_state["summerize"] = ""

    # 메인공간
    st.header(" 📽️ YouTube Summarizer ")
    st.image('ai.png', width=200)
    youtube_video_url = st.text_input("Please write down the YouTube address. 🖋️",placeholder="https://www.youtube.com/watch?v=**********")
    st.markdown('---')

    # URL을 입력 시
    if len(youtube_video_url)>2:
        if not youtube_url_check(youtube_video_url): # URL을 잘못 입력했을 경우
            st.error("YouTube URL을 확인하세요.")
        else: # URL을 제대로 입력했을 경우

            # 동영상 재생 화면 물러오기
            width = 50
            side = width/2
            _, container, _ = st.columns([side, width, side])
            container.video(data=youtube_video_url)

            # 영상 속 영어자막 추출하기
            loader = YoutubeLoader.from_youtube_url(youtube_video_url)
            transcript = loader.load()

            st.subheader("Summary Outcome (in English)")
            if st.session_state["flag"]:
                # 언어모델 설정
                llm = BedrockChat(model_kwargs={"temperature": 0},
                        model_id="anthropic.claude-v2",
                        client=bedrock
                    )
                # 프롬프트 설정
                prompt = PromptTemplate(
                    template="""백틱으로 둘러싸인 전사본을 이용해 해당 유튜브 비디오를 요약해주세요. \
                    ```{text}```
                    """, input_variables=["text"]
                )
                combine_prompt = PromptTemplate(
                    template="""Combine all the youtube video transcripts provided within backticks \
                    ```{text}```
                    Provide a concise summary between 10 to 15 sentences.
                    """, input_variables=["text"]
                )
                # LangChain을 활용하여 긴 글 요약하기
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
                text = text_splitter.split_documents(transcript)
                chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False,
                                                map_prompt=prompt, combine_prompt=combine_prompt)
                st.session_state["summerize"] = chain.run(text)
                st.session_state["flag"]=False
            st.success(st.session_state["summerize"])
            transe = trans(st.session_state["summerize"])
            st.subheader("Final Analysis Result (Reply in Korean)")
            st.info(transe)

            
if __name__=='__main__':
    main()