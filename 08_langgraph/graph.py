from typing import TypedDict
from langgraph.graph import START, StateGraph

class Essay(TypedDict):
    topic: str
    content: str | None
    score: float | None


essay = Essay(topic="This is the topic", content="This is the content", score=2.2)

def write_essay(essay: Essay):
    return{
        'content': f"The essay is about {essay['topic']}"
    }
    
def score_essay(essay: Essay):
    return{
        'score': 10.2
    }

# start -> write essay -> score essay
builder = StateGraph(Essay)

builder.add_node(write_essay)
builder.add_node(score_essay)

builder.add_edge(START, 'write_essay')
builder.add_edge('write_essay', 'score_essay')

graph = builder.compile()

state = graph.invoke({
    'topic': 'programming',
    'content': None, 
    'score': None
})

print(state)