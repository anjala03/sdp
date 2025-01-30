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
sonnet_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
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


def router_invoke(state):
    print("Invoking the router.")
    user_query = state["question"]
    # print("here form invoking the router", user_query)
    # print(state)
    return {"question": user_query}


def generate(state):
    print("state from the generator", state)
    question = state["question"]
    documents = state["documents"]
    # print("documents here", documents)
    print("Generator invoked")
    # Generate response using the retrieved documents
    context = "\n".join(documents)
    prompt = f"""Use the following pieces of retrieved context to answer the question.
<context> {context} </context>
<question> {question} </question>
Keep the answer concise."""

    generation = sonnet_llm.invoke(prompt)
    return {"generation": generation, "question": question}


def retrieve(state):
    print("Retriever invoked")
    question = state["question"]
    # print("question after retriever is invoked", question)
    docs = bedrock_retriever.invoke(question)
    # print("docs from the retriever", docs)
    documents = [doc.page_content for doc in docs]
    # print("here is the catch", documents)
    return {"documents": documents, "question": question}


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("data_source_router", router_invoke)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.set_entry_point("data_source_router")
workflow.add_edge("data_source_router", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Test case to invoke the KB
question = "What is the document about?"
graph_out = app.invoke({"question": question})
response = graph_out.get("generation").content
print(response)
