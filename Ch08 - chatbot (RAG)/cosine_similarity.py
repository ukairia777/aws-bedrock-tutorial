import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

vec1 = np.array([0,1,1,1])
vec2 = np.array([1,0,1,1])
vec3 = np.array([2,0,2,2])

print('벡터1과 벡터2의 유사도 :',cos_sim(vec1, vec2))
print('벡터1과 벡터3의 유사도 :',cos_sim(vec1, vec3))
print('벡터2와 벡터3의 유사도 :',cos_sim(vec2, vec3))