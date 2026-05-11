
from __future__ import annotations
import operator
from typing import TypedDict, List, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage



class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(..., description="One sentence describing what the reader should be able to understand after this section.")

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete, non-over lapping subpoints to cover."
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section(120-450)."
    )


    brief: str = Field(..., description="What to cover")


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str

llm = ChatOpenAI(model="gpt-4.1-mini")

def orchestrator(state: State) -> dict:
   
   plan = llm.with_structured_output(Plan).invoke(
       [
           SystemMessage(
               content=
                   "Create a blog plan with 5-7 sections on the following topic."
                   ),
           HumanMessage(content=f"Topic: {state['topic']}"),

       ]
    
   ) 
   return {"plan": plan}



def fanout(state: State ):
    return [Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
            for task in state["plan"].tasks]

def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only the section content in Markdwon."
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}

from pathlib import Path

def reducer (state: State) -> dict:
  title = state["plan"].blog_title
  body = "\n\n".join(state["sections"]).strip()

  final_md = f"# {title}\n\n{body}\n"

  filename = title.lower().replace(" ", "_") + ".md"
  output_path = Path(filename)
  output_path.write_text(final_md, encoding="utf-8")

  return {"final": final_md}

g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START,"orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile( )