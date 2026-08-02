from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-2.5-flash",temperature=0.7, max_tokens=3000)

response = model.invoke("tell me joke about urself ")

print(response.content)
