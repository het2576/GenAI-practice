from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


search_tool = TavilySearchResults(max_result = 5)

llm = ChatMistralAI(model = 'mistral-small-latest')


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("Latest AI Updates and best useful AI Tool of 2026")

result = chain.invoke({"news":news_result})

print(result)


print(search_tool.description)
print(search_tool.name)
print(search_tool.args)