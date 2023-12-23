# 패키지 불러오기
from langchain.document_loaders import YoutubeLoader

#URL 주소
youtube_video_url= "https://www.youtube.com/watch?v=ftjMM6fBRaw"
#유튜브 스크립트 추출하기
loader = YoutubeLoader.from_youtube_url(youtube_video_url)
transcript = loader.load()
print(transcript)