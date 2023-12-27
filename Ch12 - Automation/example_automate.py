import pandas as pd
import boto3
import json
from tqdm import tqdm
import ast

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

df = pd.read_excel('./example.xlsx')

sub_df = df[df['고장부품'].isnull()]
sub_df.reset_index(drop=True, inplace=True)

def ask_claude(prompt):
    body = json.dumps({
        "prompt": "\n\nHuman: " + prompt + "\n\nAssistant:",
        "max_tokens_to_sample": 300,
        "temperature": 0,
    })

    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())

    # text
    return response_body.get('completion')

def make_prompt(text):
    prompt = '''주어진 고장내용 텍스트로부터 고장부품, 불량유형, 조치사항을 json 형태로 추출하시오.
    예시는 다음과 같습니다.
    
    고장내용: '출하장 입구 에어 50A 유니온 에어리크 재조임'
    출력: {"고장부품": "50A 유니온", "불량유형" : "에어리크", "조치내용" : "재조임"}

    고장내용: '출하장 입구 에어 50A 유니온 에어리크 재조임'
    출력: {"고장부품": "SHIELD ROOM 화풍기", "불량유형" : "철거부위 마감", "조치내용" : "판넬벽체문타공"}

    자, 실전입니다. 아래의 고장내용을 json 형태로 추출해보세요.
    
    고장내용: ''' + str(text) + '''
    '''
    return prompt

result = []
for sample in tqdm(sub_df['고장내용'].to_list()):
    prompt = make_prompt(sample)
    result.append(ask_claude(prompt))

# 앞 공백이나 줄바꿈 제거
result = [r.strip().replace('\n', '') for r in result]

# 원소의 type을 문자열에서 dict로 변환.
result = [ast.literal_eval(elem) for elem in result]

new_df = pd.DataFrame(result)
sub_df['고장부품'] = new_df['고장부품']
sub_df['불량유형'] = new_df['불량유형']
sub_df['조치내용'] = new_df['조치내용']

sub_df.to_excel('new_data.xlsx', index=False)