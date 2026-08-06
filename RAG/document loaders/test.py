from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TextSplitter

data = TextLoader("document loaders/notes.txt")

print(data)
docs = data.load()
print(docs[0])