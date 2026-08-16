from dotenv import load_dotenv
import os
import requests

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call
from datetime import datetime
import pytz


@tool
def get_websearch(query: str) -> str:
    """Search the web and return the top 3 result links."""

    response = tavily_client.search(

        query=query,
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])
    if not results:
        return "No web results found."
    
    return "\n\n".join(
        f"{i + 1}. {result.get('title', 'No title')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Content: {result.get('content', '')[:500]}"
        for i, result in enumerate(results)
    )
   
    

@tool

def get_time(zone: str) -> str:

    """Get the current time for a given IANA timezone."""
    time_zone = pytz.timezone(zone)
    current_time = datetime.now(time_zone)
    return f"Time in {zone}: {current_time.strftime('%H:%M:%S')}"

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"


    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


llm = ChatMistralAI(model="mistral-small-latest")

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  



agent = create_agent(
    llm,
    tools = [get_weather,get_news,get_time,get_websearch],
    system_prompt= "you are a helpful city assistant.",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("bot : ", result['messages'][-1].content )