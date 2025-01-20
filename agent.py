from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_aws import ChatBedrock, BedrockLLM
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate
from typing import TypedDict, Dict, List
from langgraph.checkpoint.memory import MemorySaver

model_id = "anthropic.claude-v2:1"
llm = BedrockLLM(
    model_id=model_id
)
print("llm", llm)


def use_case_generator(state: MessagesState):
    prompt_template = """You are an advanced SDP use case generator specialized in AWS services.
    Your role is to generate specific, practical use cases for AWS services specifically for RDS based on their descriptions.
    Focus on real-world applications and business value.
    Each use case should include:
    1. Scenario description
    2. Business challenge
    3. Implementation approach
    4. Expected benefits
    Maintain context from previous interactions and build upon existing use cases when appropriate.
    Use the provided documents as the references:
    """
    message = SystemMessage(content=prompt_template)
    output = llm.invoke([message]+state["messages"])
    print("here is the response", output)
    return {"messages": state["messages"]}


def document_generator(state: MessagesState):
    prompt_template = """You are a technical aws sdp documentation specialist.
    Based on the use cases provided in the previous state, create comprehensive sdp technical documentation.
    Include the following sections:
    1. Overview
    2. Architecture Design
    3. Implementation Steps
    4. Configuration Details
    5. Best Practices
    6. Monitoring and Maintenance
    Format the documentation in a clear, professional structure suitable for technical teams."""

    message = SystemMessage(content=prompt_template)
    output = llm.invoke([message] + state["messages"])
    return {"messages": output}


builder = StateGraph(MessagesState)
builder.add_node("use_case_agent", use_case_generator)
builder.add_node("document_generator_agent", document_generator)
builder.add_edge(START, "use_case_agent")
builder.add_edge("use_case_agent", "document_generator_agent")
builder.add_edge("document_generator_agent", END)


memory = MemorySaver()
# Compile graph
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

with open("rds.txt", "r") as file:
    documents = file.read()

response = graph.invoke({"messages": HumanMessage(content=f"provide sdp relevant documentations.Here is the references {documents}")}, config)
generated_message = response["messages"][1].content
with open("test.txt", "w") as output:
    output.write(generated_message)
