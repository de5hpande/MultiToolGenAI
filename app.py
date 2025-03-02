import streamlit as st
import os
import cassio
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import END, StateGraph, START
from typing import List, Literal
from typing_extensions import TypedDict


# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
groq_api_key = os.getenv("groq_api_key")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Streamlit UI
st.title("MultiTool AI RAG")

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)
astra_vector_store.add_documents(doc_splits)
st.success(f"Inserted {len(doc_splits)} documents into vector store.")

retriever = astra_vector_store.as_retriever()

# Routing System
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"]

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

route_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert router for vectorstore or Wikipedia."),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# Wiki and Arxiv
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))

# Graph Definition
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

def wiki_search(state):
    docs = wiki.invoke({"query": state["question"]})
    return {"documents": [docs], "question": state["question"]}

def route_question(state):
    source = question_router.invoke({"question": state["question"]})
    return "wiki_search" if source.datasource == "wiki_search" else "retrieve"

workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(START, route_question, {"wiki_search": "wiki_search", "vectorstore": "retrieve"})
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
app = workflow.compile()

# Streamlit UI Query
user_input = st.text_area("Enter your query:")
if st.button("Submit"):
    if user_input:
        st.spinner("Generating response...")
        inputs = {"question": user_input}
        for output in app.stream(inputs):
            for key, value in output.items():
                st.write(f"**{key}:** {value}")
    else:
        st.warning("Please enter a query.")
