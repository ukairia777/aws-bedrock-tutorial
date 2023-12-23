from openai import OpenAI
import streamlit as st
import boto3
import json

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)


def trans(q, option):
    body = json.dumps({
        "prompt" : f"""\n\nHuman: "You are a helpful translator. Please do not write any sentences other than the translation result. Translate the text sent by the user into {option}"
        입력: "{q}" \n\nAssistant: """,
        "max_tokens_to_sample": 1000,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    print(body)
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    return response_body.get('completion')

lang_list = ('영어', '중국어', '일본어', '한국어')

st.header('Translator 💌')
st.image('ai.png', width=200)

col1, col2 = st.columns(2)
result = ''
with col1:
    option = st.selectbox('Target Language', lang_list)
    q = st.text_area('From')
    response = ''
    if q:
        print('출력:', q)
        response = trans(q, option)
        print(response)

with col2:
    st.text_area('To', value=response)