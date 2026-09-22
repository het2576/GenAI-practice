from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. TOOLS
# ============================================================

search_tool = TavilySearchResults(max_results=3)

tools = [search_tool]


# ============================================================
# 3. LLMs
# ============================================================

# Writer LLM
writer_llm = ChatMistralAI(
    model="mistral-medium-latest",
    temperature=0.7
)

# Give the writer access to tools
writer_llm_with_tools = writer_llm.bind_tools(tools)


# Reviewer LLM
reviewer_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2
)


# ============================================================
# 4. STATE
# ============================================================

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ============================================================
# 5. WRITER PROMPT
# ============================================================

WRITER_SYSTEM_PROMPT = """
You are an expert LinkedIn content writer.

Your job is to write engaging, professional LinkedIn posts
about the given topic.

If the topic requires up-to-date information, statistics,
or current trends, use the web search tool before writing.

If you receive feedback from a previous review, carefully
address every issue mentioned in the feedback.

Rules for good LinkedIn posts:

1. Strong hook in the first line.
2. One clear and valuable takeaway.
3. Easy to skim with short paragraphs.
4. Around 150-200 words.
5. End with a question or call-to-action.
6. Professional but human tone.
7. Do not use hashtags.

Return only the LinkedIn post when you are finished.
"""


# ============================================================
# 6. WRITER NODE
# ============================================================

def writer_node(state: State) -> dict:
    """
    Writes a new LinkedIn post or continues writing after
    receiving results from a tool.
    """

    topic = state["topic"]
    current_attempt = state.get("attempt", 0)

    # Existing messages in the graph
    existing_messages = state.get("messages", [])

    # --------------------------------------------------------
    # CASE 1:
    # The previous message is a tool result.
    #
    # The writer should NOT start a new attempt.
    # It should simply use the search results and finish
    # writing the post.
    # --------------------------------------------------------

    if existing_messages and existing_messages[-1].type == "tool":

        messages = [
            ("system", WRITER_SYSTEM_PROMPT),
            *existing_messages,
            (
                "human",
                "Use the search results above to write the final "
                "LinkedIn post. Return only the post."
            ),
        ]

        response = writer_llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    # --------------------------------------------------------
    # CASE 2:
    # New attempt
    # --------------------------------------------------------

    attempt = current_attempt + 1

    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:

        user_message = (
            f"Write a LinkedIn post about this topic:\n\n"
            f"{topic}\n\n"
            f"If current information is required, use the web "
            f"search tool first."
        )

    else:

        user_message = (
            f"Your previous LinkedIn post about '{topic}' "
            f"was rejected.\n\n"
            f"Here is the reviewer's feedback:\n\n"
            f"{previous_feedback}\n\n"
            f"Write a new and improved version that fixes "
            f"every issue mentioned above.\n"
            f"Do not repeat the same mistakes."
        )

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        *existing_messages,
        ("human", user_message),
    ]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "attempt": attempt
    }


# ============================================================
# 7. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 8. EXTRACT DRAFT NODE
# ============================================================

def extract_draft_node(state: State) -> dict:
    """
    Extracts the final text generated by the writer
    and stores it as the current draft.
    """

    last_message = state["messages"][-1]

    draft = last_message.content.strip()

    print("\n" + "=" * 55)
    print("GENERATED DRAFT")
    print("=" * 55)
    print(draft)
    print("=" * 55)

    return {
        "draft": draft
    }


# ============================================================
# 9. REVIEWER PROMPT
# ============================================================

REVIEWER_SYSTEM_PROMPT = """
You are a strict LinkedIn content reviewer.

Your job is to decide whether a LinkedIn post is
publish-ready.

Evaluate the post using these criteria:

1. Strong hook in the first line.
2. One clear and valuable takeaway.
3. Easy to skim with short paragraphs.
4. Roughly 150-200 words.
5. Ends with an engaging question or CTA.
6. Professional but human tone.
7. No hashtags.

Respond EXACTLY in this format:

VERDICT: APPROVED
FEEDBACK: <one short paragraph>

OR

VERDICT: REJECTED
FEEDBACK: <one short paragraph>

Be strict but fair.
Approve only if the post genuinely satisfies all criteria.
"""


# ============================================================
# 10. REVIEWER NODE
# ============================================================

def reviewer_node(state: State) -> dict:
    """
    Reviews the current draft and decides whether it
    should be approved or rewritten.
    """

    draft = state["draft"]

    prompt = (
        "Review the following LinkedIn post draft:\n\n"
        f"{draft}\n\n"
        "Evaluate it using all the criteria provided in "
        "your system instructions."
    )

    response = reviewer_llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", prompt)
        ]
    )

    review_text = response.content.strip()

    # --------------------------------------------------------
    # Extract verdict
    # --------------------------------------------------------

    verdict_part = review_text.upper().split("FEEDBACK:", 1)[0]

    is_approved = "VERDICT: APPROVED" in verdict_part

    # --------------------------------------------------------
    # Extract feedback
    # --------------------------------------------------------

    if "FEEDBACK:" in review_text:
        feedback = review_text.split(
            "FEEDBACK:", 1
        )[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"

    print(f"\n[Reviewer Verdict: {verdict}]")
    print(f"[Reviewer Feedback: {feedback}]\n")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# ============================================================
# 11. ROUTER:
#     SHOULD THE WRITER USE A TOOL?
# ============================================================

def should_use_tool(state: State):
    """
    Checks whether the writer requested a tool.

    YES  -> execute the tool
    NO   -> extract the draft
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "extract_draft"


# ============================================================
# 12. ROUTER:
#     SHOULD WE CONTINUE OR STOP?
# ============================================================

def should_stop_looping(state: State):
    """
    Decides whether the post is finished or should
    be rewritten.
    """

    # Reviewer approved
    if state["is_approved"]:
        print("\nPost has been approved.")
        return END

    # Maximum 3 writing attempts
    if state["attempt"] >= 3:
        print("\nReached maximum attempts.")
        return END

    # Reviewer rejected -> send back to writer
    print("\nSending feedback back to writer...")
    return "writer"


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)


# Add nodes
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)


# ------------------------------------------------------------
# START → WRITER
# ------------------------------------------------------------

graph.add_edge(
    START,
    "writer"
)


# ------------------------------------------------------------
# WRITER → TOOL or EXTRACT
# ------------------------------------------------------------

graph.add_conditional_edges(
    "writer",
    should_use_tool
)


# ------------------------------------------------------------
# IMPORTANT:
#
# TOOL → WRITER
#
# After Tavily gives results, the writer gets those
# results and finishes the post.
# ------------------------------------------------------------

graph.add_edge(
    "tools",
    "writer"
)


# ------------------------------------------------------------
# EXTRACT → REVIEWER
# ------------------------------------------------------------

graph.add_edge(
    "extract_draft",
    "reviewer"
)


# ------------------------------------------------------------
# REVIEWER → END or WRITER
# ------------------------------------------------------------

graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)


# Compile graph
app = graph.compile()


# ============================================================
# 14. USER INTERFACE
# ============================================================

print("=" * 55)
print("WELCOME TO THE LINKEDIN POST GENERATOR")
print("=" * 55)

print(
    "\nThis agent will:"
    "\n1. Write a LinkedIn post"
    "\n2. Search the web if needed"
    "\n3. Review the post"
    "\n4. Rewrite it if rejected"
    "\n5. Stop when approved or after 3 attempts"
)

print("=" * 55)


topic = input(
    "\nWhat topic do you want a LinkedIn post about?\n> "
).strip()


if not topic:

    print("\nNo topic given. Exiting.")

else:

    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)

    print(final_state["draft"])

    print("=" * 55)

    print(
        f"Total attempts: {final_state['attempt']}"
    )

    print(
        f"Approved: {final_state['is_approved']}"
    )