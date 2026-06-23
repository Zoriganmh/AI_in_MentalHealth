from google import genai

# Điền API Key của bạn vào đây
client = genai.Client(api_key="AIzaSyC2qo2nimZLAcqpeB9TUHpl2t-PK4_PBWY")

print("Danh sách các model bạn có thể dùng:")
for model in client.models.list():
    print(model.name)