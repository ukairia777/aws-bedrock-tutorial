import streamlit as st
import requests
import pandas as pd
import os
import io
import boto3 
import json
import base64
from io import BytesIO
from PIL import Image

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
) 

bedrock_model_id = "stability.stable-diffusion-xl-v0"

def get_response_image_from_payload(response): 

    payload = json.loads(response.get('body').read()) 
    images = payload.get('artifacts')
    image_data = base64.b64decode(images[0].get('base64'))

    return BytesIO(image_data) 

def get_image_response(prompt_content): 
    
    request_body = json.dumps({"text_prompts": 
                               [ {"text": prompt_content } ], 
                               "cfg_scale": 9, 
                               "steps": 50, }) 
    
    response = bedrock.invoke_model(body=request_body, modelId=bedrock_model_id)
    
    output = get_response_image_from_payload(response) 
     
    return output

st.title("그림 그리는 AI 화가 서비스 👨‍🎨")

st.image('https://wikidocs.net/images/page/215361/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%ED%99%94%EA%B0%80.png', width=200)

st.text("🎨 Tell me the picture you want. I'll draw it for you!")

input_text = st.text_area("원하는 이미지의 설명을 영어로 적어보세요.", height=200)
if st.button("Painting"):
    if input_text:
        try:
            summary = get_image_response(input_text)
            st.image(summary)
        except:
            st.error("요청 오류가 발생했습니다")
    else:
        st.warning("텍스트를 입력하세요")