from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

model = ChatMistralAI(model= 'mistral-small-latest')
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a AI who generate the code"),
    ("human" , "generate the code of {topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assitant who explains the code in simplest term step by step"),
    ("human" , "explain {topic} code")
])

seq1 = code_prompt | model | parser

seq2 = RunnableParallel({
    "code" : RunnablePassthrough(),
    "explanation": explain_prompt | model | parser
})

chain  = seq1 | seq2 

result = chain.invoke({"topic" : "please write a code of palindrome in python "})

print(result['code'])
print(result['explanation'])