import uvicorn
import os
import logging 
from fastapi import FastAPI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model


load_dotenv()

app = FastAPI()


def setup_chain():
    
    # read in pdf file
    pdf_reader = PdfReader("/app/data/goss_web_content_cleaned.pdf")
    # read data from the file and put them into a variable called text
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # Split text file into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, length_function=len)
    text_chunks = text_splitter.split_text(text)

    # Define LLM and Embedding Model
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 400,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.TEMPERATURE: 0.1,
        GenParams.RANDOM_SEED: 42,
        GenParams.STOP_SEQUENCES: ["\n\n"]}
        
    llama = Model(
        model_id=ModelTypes.FLAN_UL2, 
        params=parameters, 
        credentials={
            "apikey": os.environ["IBM_CLOUD_API_KEY"],
            "url": os.environ["IBM_CLOUD_URL"]},
        project_id=os.environ["WATSONX_PROJECT_ID"])
    
    llm = WatsonxLLM(model=llama)
    
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'})
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    
    # Setup Prompt
    system_prompt = """
                    You are an expert AI Q&A system that is trusted around the world. 

                    Your tone should be professional and natural, as human response.
                                                
                    Context information is below.
                    ----------------------------
                    {context}
                    ----------------------------

                    Answer the user's questions only based on given context information and no prior knowledge.

                    Avoid statements like 'Based on the context, ...' or 'Based on the provided document, ...' or 'According to the provided context ...', or anything along those lines.
                                                
                    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                                                
                    If the user asks questions that are not related to the retrieved information, respond with: "I’m sorry, but I’m not certain about the answer to your question. Could you kindly provide another question pertaining to the company?" and stop after that.

                    If the answer is not included, respond with: "I’m sorry, but I’m not sure about the answer to your question. Please feel free to ask another question and thank you for your understanding." and stop after that.
                    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
     
    # Conversational Chain 
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message_prompt,
                    human_message_prompt,
                ]
            ),
        }
    )
    
    return conversation_chain
    
# Initialize agent once on startup 
agent = setup_chain()

# Starting the chain setup only once on startup

'''
The @app.on_event("startup") decorator in FastAPI is used to run a function when the server starts up.
@app.on_event("startup") runs on server start
It's used to initialize the agent one time on startup
This avoids having to reload the agent on every request
The agent will be ready to use when requests start coming in
'''
  
@app.on_event("startup") 
async def startup():
    print("Setting up agent...")
    agent = setup_chain() 
    
@app.get("/")
def home():
  return "Welcome to RAG Chatbot API!" 

@app.get("/api/chat")
async def chat(question: str):
    print(f"Question: {question}")
    response = agent.invoke(question)
    return {"Answer": response["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
        

    
