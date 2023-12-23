# pip install pypdf==3.17.3
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


# get_rag_chat_response()에서 호출.
def get_llm():
    model_kwargs = {
        "temperature": 0,
        "max_tokens_to_sample": 1024,
    }
    
    llm = Bedrock(
        region_name='us-east-1',
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        model_id="anthropic.claude-v2",
        model_kwargs=model_kwargs)
    
    return llm

def get_index(): 
    embeddings = BedrockEmbeddings(
        region_name='us-east-1',
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        model_id="amazon.titan-embed-text-v1"
    ) 
    
    pdf_path = "./2022-Shareholder-Letter.pdf"
    loader = PyPDFLoader(file_path=pdf_path) 
    
    text_splitter = RecursiveCharacterTextSplitter( 
        separators=["\n\n", "\n", ".", " "], 
        chunk_size=1000, 
        chunk_overlap=100 
    )
    
    index_creator = VectorstoreIndexCreator( 
        vectorstore_cls=FAISS, 
        embedding=embeddings,
        text_splitter=text_splitter,
    )
    
    index_from_loader = index_creator.from_loaders([loader])
    
    return index_from_loader

def get_memory():
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)  
    return memory

def get_rag_chat_response(input_text, memory, index):
    llm = get_llm()
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory) 
    chat_response = conversation_with_retrieval({"question": input_text})  
    print(chat_response)
    return chat_response['answer']