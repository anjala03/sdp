import boto3
from botocore.config import Config
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws import ChatBedrock
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict

# Setup boto3 config to allow for retrying
region = "us-east-1"
my_config = Config(
    region_name=region,
    signature_version='v4',
    retries={
        'max_attempts': 3,
        'mode': 'standard'
    }
)

# Setup bedrock runtime client
bedrock_rt = boto3.client("bedrock-runtime", config=my_config)

# Setup LLM
sonnet_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["Human"],
}

sonnet_llm = ChatBedrock(
    client=bedrock_rt,
    model_id=sonnet_model_id,
    model_kwargs=model_kwargs,
)

# Setup retriever
bedrock_retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="O4JIGBKU1Y",
    region_name=region,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)


class GraphState(TypedDict):
    question: str
    documents: str
    generation: str
    question_type: str


def router_invoke(state):
    print("Invoking the router.")
    user_query = state["question"]
    return {"question": user_query}


def query_router(state):
    print("Analyzing the query whether it is general or domain based.")
    prompt = f"""
Analyze the following question: {state["question"]}.

If the question pertains to a specialized domain such as AWS, Azure, service delivery programs, or other technical/business-specific services, return exactly: **domain-specific**.

If the question is general and does not belong to a specialized domain, return exactly: **general**.

Your response should be only one of these two words: "domain-specific" or "general", with no additional text.
"""

    response = sonnet_llm.invoke(prompt)
    print('here is the question type:', response.content)
    return {"question": question, "question_type": response.content}


def generate(state):
    question = state["question"]
    question_type = state["question_type"]
    if "domain-specific" not in question_type:
        generated_response = sonnet_llm.invoke(question)
    else:
        documents = state["documents"]
        print("Generator invoked")
        context = "\n".join(documents)
        prompt = f"""Use the following pieces of retrieved context to answer the question.
        <context> {context} </context>
        <question> {question} </question>
        Keep the answer concise.If you dont know anything dont provide random output just say I dont have information regarding this concept."""
        generated_response = sonnet_llm.invoke(prompt)
    return {"generation": generated_response, "question": question}


def retrieve(state):
    print("Retriever invoked")
    question = state["question"]
    docs = bedrock_retriever.invoke(question)
    documents = [doc.page_content for doc in docs]
    return {"documents": documents, "question": question}


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("query_router", query_router)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.set_entry_point("query_router")
workflow.add_edge("query_router", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Test case to invoke the KB
question = "what is aws sdp program?."
graph_out = app.invoke({"question": question})
response = graph_out.get("generation").content
print(response)
