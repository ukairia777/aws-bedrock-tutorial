from openai import OpenAI

audio_file = open("./output.mp3", "rb")

client = OpenAI(
    api_key = '여러분들의 OpenAI Key값'
)

transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
print(transcript)