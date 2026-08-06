from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = [
    "Hello there how are you?",
    "My name is het",
    "I am learning the GenAI"
]


embeddings= GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

vectors = embeddings.embed_documents(text)
print(vectors)