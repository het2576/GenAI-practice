from langchain_core.tools import tool

@tool
def greetings(name: str) -> str : 
    """Generate the greeting msg for the users"""

    return f"hello {name} welcome to the AI world"

res = greetings.invoke({"name": "Het"})

print(res)

print(greetings.args)
print(greetings.description)
