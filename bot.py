# from dotenv import load_dotenv
import os
import openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY
OPENAI_API_KEY='Your_OpenAI_API'
# Initialize FAISS and Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=OPENAI_API_KEY)
vector_store = FAISS.load_local(
    "faiss_store", embeddings, allow_dangerous_deserialization=True
)

HR_PROMPT = """\
You are an AI assistant for the question ans. Your role is to provide accurate and concise answers
Use the following context to answer the user's question. 
if there is calculations performe it accurately and provide accurate answers.

Question: {question}

Relevant Information: {context}

"""

def get_docs(query, top_k=5):
    results = vector_store.similarity_search(
        query=query,
        k=top_k,
    )
    formated_docs = ""
    
    for result in results:
        formated_docs += result.page_content + "\n"
    
    return formated_docs

def run(query):
    docs = get_docs(query)

    prompt = PromptTemplate(
        input_variables=["question", "context"], template=HR_PROMPT
    )
    llm = ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)
    chain = prompt | llm | StrOutputParser()
    data = {
        "question": query,
        "context": docs
    }

    response = chain.invoke(data)
    
    return response
