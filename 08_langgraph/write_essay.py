import dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from dataclasses import dataclass


class ContentState(TypedDict):
    topic: str  # Topic of the essay
    draft: str  # The most recent version of essay written by first LLM
    score: int  # Score assigned by another LLM
    feedback: str  # Feedback by LLM about what needs to be approved on the essay
    revision_count: int  # Count about how many times this essay has been revised. It is used to break the iteration.


@dataclass
class Context:
    llm: object


def write_draft(state: ContentState, runtime: Runtime[Context]):
    llm = runtime.context.llm
    prompt = f"Write a short paragraph about: {state['topic']}"

    if state.get("feedback"):
        prompt += f"\n\nPrevious feedback to address: {state['feedback']}"

    response = llm.invoke(prompt)

    return {"draft": response.content, "revision_count": state["revision_count"] + 1}


def review_draft(state: ContentState, runtime: Runtime[Context]):
    llm = runtime.context.llm
    prompt = f"""Rate this paragraph from 1-10 and give one sentence of feedback.
    Paragraph: {state["draft"]}
    
    Respond in exactly this format: 
    Score: <number>
    Feedback: <text>"""

    response = llm.invoke(prompt)

    text: str = response.content

    score = 0
    feedback = ""
    for line in text.split("\n"):
        if line.startswith("Score"):
            score = int(line[6:])
            feedback = line[9:]

    return {"score": score, "feedback": feedback}


def should_revise(state: ContentState):
    if state["score"] >= 8 or state["revision_count"] >= 3:
        return END
    return "write_draft"


if __name__ == "__main__":
    dotenv.load_dotenv()

    model = "openai/gpt-oss-20b"
    temperature = 0.7
    llm = ChatGroq(model=model, temperature=temperature)

    builder = StateGraph(ContentState, context_schema=Context)

    builder.add_node("write_draft", write_draft)
    builder.add_node("review_draft", review_draft)

    builder.add_edge(START, "write_draft")
    builder.add_edge("write_draft", "review_draft")
    builder.add_conditional_edges("review_draft", should_revise)

    graph = builder.compile()
    result = graph.invoke(
        {
            "topic": "why sleep matters for learning",
            "draft": "",
            "score": 0,
            "feedback": "",
            "revision_count": 0.0,
        },
        context={"llm": llm},
    )

    print(f"Final draft ({result['revision_count']}) revision(s): \n")
    print(result["draft"])
    print(f"\nFinal score: {result['score']}")
