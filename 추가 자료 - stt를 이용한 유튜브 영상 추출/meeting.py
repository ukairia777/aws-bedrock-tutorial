import streamlit as st
import os 
import openai
from moviepy.editor import VideoFileClip, AudioFileClip
import math
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

def split_file(filename, interval=120):
    # 분할된 파일명을 저장할 리스트
    split_filenames = []

    # 파일 확장자에 따라 오디오 또는 비디오 클립을 로드합니다.
    if filename.endswith(('.mp3', '.m4a', '.wav', '.mpga')):
        clip = AudioFileClip(filename)
    else:
        clip = VideoFileClip(filename)
    
    # 파일을 자를 총 부분의 수를 계산합니다.
    total_parts = math.ceil(clip.duration / interval)
    st.write('2분 단위로 영상을 나누기 위해서 총 ' + str(total_parts) + '개의 영상으로 분할해야 합니다.')

    for part in range(total_parts):
        # 시작 및 종료 시간을 계산합니다.
        start_time = part * interval
        end_time = min((part + 1) * interval, clip.duration)

        # 클립을 자릅니다.
        new_clip = clip.subclip(start_time, end_time)

        # 새 파일 이름을 생성합니다.
        new_filename = f"{filename.rsplit('.', 1)[0]}_part{part + 1}.{filename.rsplit('.', 1)[1]}"
        split_filenames.append(new_filename)

        # 새 파일을 저장합니다.
        if not filename.endswith(('.mp3', '.m4a', '.wav', '.mpga')):
            new_clip.write_videofile(new_filename, codec="libx264")
        else:
            new_clip.write_audiofile(new_filename)

        st.write(new_filename + ' 파일을 저장했습니다.')

    clip.close()

    # 분할된 파일명 리스트를 반환합니다.
    return split_filenames

def summarize(client, text):
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following in 5 bullet points in korean and in formal language"},
                    {"role": "user", "content": text}
                ]
            )
    return response.choices[0].message.content


def main():

    client = openai.OpenAI(
    api_key = '여러분들의 OpenAI Key값'
    )

    st.set_page_config(page_title="회의록을 작성하는 인공지능")
    st.session_state.setdefault("audio_file_path", None)
    st.session_state.setdefault("transcript", None)
    st.title("Meeting Minutes AI 🖋️")
    st.image('meeting.png', width=300)

    uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'mp4', 'mpeg', 'mpga', 
                                                                'm4a', 'wav', 'webm'])

    if st.button("Generate Meeting Minutes") and uploaded_file:
        with st.spinner('Processing...'):
            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_file_path = file_path
            split_filenames = split_file(file_path)

            result = ''

            for sub_file in split_filenames:
                with open(sub_file, 'rb') as audio_file:
                    st.write(sub_file, '을 분석합니다.')
                    st.session_state.transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                response_format="text"
                                )
                result += st.session_state.transcript

            file_path = "./youtube_text.txt"
            with open(file_path, 'w') as file:
                # 파일에 문자열을 씁니다.
                file.write(result)

            st.write('영상의 내용을 모두 분석하였습니다. 전체 내용을 현재 경로에 youtube_text.txt 파일로 저장합니다. 이제 회의록 작성을 위해 텍스트를 분할합니다.')
            loader = DirectoryLoader('.', glob="*.txt", loader_cls=TextLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            st.write('분할된 텍스트의 개수 :', len(texts))

            final_result = ''
            for t in texts:
                st.write('회의록을 작성 중...')
                final_result += summarize(client, str(t))

    if st.session_state.audio_file_path:
        st.subheader("회의록")
        st.write(st.session_state.audio_file_path.split("\\")[1])
        if result:
            st.markdown(final_result)

if __name__ == "__main__":
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    main()