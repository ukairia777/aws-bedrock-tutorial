# Langchain 패키지들
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import boto3
import json

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

script = ''' 텍스트 입력'''

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
# 글 쪼개기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
texts = text_splitter.create_documents([script])
# 요약하기
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False,
                            map_prompt=prompt, combine_prompt=combine_prompt)
summerize = chain.run(texts)
# 최종 출력
print(summerize)