import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# # model checkpoint
# checkpoint = "LaMini-T5-738M"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map="auto",
#     torch_dtype=torch.float32,
# )

# define embeddings
@st.cache_resource
def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# # pipeline
# @st.cache_resource
# def get_llm():
#     pipe = pipeline(
#         task="text2text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=512,
#         temperature=0.3,
#         top_p=0.9,
#     )
#     local_llm = HuggingFacePipeline(pipeline=pipe)
#     return local_llm

class get_llm:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_requested_llm(self):
        if self.model_name == 'OpenAI':
            llm = ChatOpenAI()
        elif self.model_name == 'llama2':
            llm = Ollama(model="llama2")
        return llm
