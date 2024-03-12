import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from get_embeddings_and_models import get_embeddings, get_llm
from dotenv import load_dotenv

load_dotenv()

# load the API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def get_vectorstore(url, embeddings):
    """
    Scrapes the website, chunks the document and creates a vectorstore from the chunks.

    Args:
        url (str): URL of the website to scrape
        embeddings : Embedding model
    
    Returns:
        vectorstore (VectorStore): VectorStore created from the chunks of the website
    """
    # scrape the website
    loader = WebBaseLoader(url)
    document = loader.load()
    # chunk the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(document)
    # get vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
    
def get_retriever_chain(vectorstore, llm):
    """
    Creates a history aware retriever chain from the vectorstore.
    
    Args:
        vectorstore (VectorStore): VectorStore created from the chunks of the website
        llm : llm model
        
    Returns:
        retriever_chain: History aware retriever chain
    """
    # llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_stuff_documents_chain(retriever_chain, llm):
    """
    Creates a stuff documents chain from the retriever chain.
    
    Args:
        retriever_chain: History aware retriever chain
        llm : llm model
        
    Returns:
        stuff_documents_chain: Stuff documents chain
    """
    # llm = ChatOpenAI()
    # creating prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's query based on the following context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # create the stuff documents chain
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    # create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

    return retrieval_chain

# Page Config
st.set_page_config(page_title="chat with website", page_icon="🤖")

st.title("Chat with Website")
# sidebar
with st.sidebar:
    if "model" not in st.session_state:
        st.session_state.model = "None"
    models = ["OpenAI", "llama2"]
    if st.session_state.model == "None":
        st.write("Please select a model")
        model = st.selectbox("Select the model", models)
    
    st.write(f"Selected model: {model}")
    st.header("Web URL")
    website = st.text_input("Enter a valid URL...")

if website is None or website == "":
    st.info("Please enter URL to get started.")

else:
    # chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    if model=="OpenAI":
        embeddings = OpenAIEmbeddings()
        llm = get_llm(model_name="OpenAI").get_requested_llm()
    elif model=="llama2":
        embeddings = get_embeddings()
        llm = get_llm(model_name="llama2").get_requested_llm()
    
    if "vectorstore" not in st.session_state:
        # get vectorstore
        st.session_state.vectorstore = get_vectorstore(website, embeddings=embeddings)

    # get retriever chain
    retriever_chain = get_retriever_chain(st.session_state.vectorstore, llm=llm)

    # get stuff documents chain
    stuff_documents_chain = get_stuff_documents_chain(retriever_chain, llm=llm)

    # user input
    user_query = st.chat_input("Enter your Query...")

    if user_query is not None and user_query != "":
        response = stuff_documents_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query,
        })
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response['answer']))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👤"):
                st.write(message.content)
