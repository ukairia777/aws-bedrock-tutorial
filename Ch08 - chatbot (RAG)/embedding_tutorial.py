
import json
import boto3
import math
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

def get_embedding(text):
    body = json.dumps({"inputText": text})
    model_d = 'amazon.titan-embed-text-v1'
    mime_type = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=model_d, accept=mime_type, contentType=mime_type)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

embedding_result = get_embedding('저는 배가 고파요')

data = ['저는 배가 고파요',
        '저기 배가 지나가네요',
        '굶어서 허기가 지네요',
        '허기 워기라는 게임이 있는데 즐거워',
        '스팀에서 재밌는 거 해야지',
        '스팀에어프라이어로 연어구이 해먹을거야']

df = pd.DataFrame(data, columns=['text'])
df['embedding'] = df.apply(lambda row: get_embedding(
        row.text,
    ), axis=1)

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query,
    )
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x),
                                                            np.array(query_embedding)))
    results_co = df.sort_values("similarity",
                                ascending=False,
                                ignore_index=True).iloc[:3, :]
    return results_co.head(3)

sim_result = return_answer_candidate(df, '아무 것도 안 먹었더니 꼬르륵 소리가나네')
print(sim_result)