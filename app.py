import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OLLAMA"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Initialize LLM model once
llm_model = None

def initialize_llm(model_name):
    global llm_model
    if not llm_model:
        llm_model = Ollama(model=model_name)
    return llm_model

def generate_response(question, llm_name, temperature):
    # Initialize LLM
    llm = initialize_llm(llm_name)

    # Setup output parser
    output_parser = StrOutputParser()

    # Create chain
    chain = prompt | llm | output_parser

    # Generate answer
    try:
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit app
st.title("Enhanced Q&A Chatbot with Ollama")

# Dropdown to select models
llm = st.sidebar.selectbox("Select an Open Source model", ["llama3.2"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)


# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature)
    st.write(response)
else:
    st.write("Please provide the user input")
