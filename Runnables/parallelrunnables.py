from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

model = ChatMistralAI(model="mistral-small-latest")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
      "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
      "Explain {topic} in detailed pointwise"
)

topic = "Machine Learning"

chain = RunnableParallel({

        "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
        "detailed": RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | parser,
    }
)

response = chain.invoke({
    "short" : {"topic":"Machine Learning"},
    "detailed" : {"topic":"Deep Learning"}
})

print (response['short'])
print (response['detailed'])
