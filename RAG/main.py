from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = MistralAIEmbeddings(model='mistral-embed')

vectorstore = Chroma(
    embedding_function= embeddings,
    persist_directory='chroma_db'
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
      search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

llm = ChatMistralAI(model="mistral-small-latest")

prompt = ChatPromptTemplate(
    [
      ("system",
   """
      You are a helpful AI assistant.

      Use ONLY the provided context to answer the question.

      If the answer is not present in the context,
      say: "I could not find the answer in the document.

   """),
   ("human",
   """ 
   Context:
   {context}

   Question:
   {question}
   """
   )
   ]
)


print("RAG System created!")

print("press 0 to exit ")


while True:
    query = input("You :")

    if query == "0":
        break
    
    docs = retriever.invoke(query)

    context =  "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
      "context" :context,
      "question": query
    })

    response = llm.invoke(final_prompt)

    print(response)
   