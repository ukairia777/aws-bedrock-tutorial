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
    '''  
    llm = get_llm()
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory) 
    chat_response = conversation_with_retrieval({"question": input_text + '한국어로 답변해줘'})  
    print(chat_response)
    '''
    '''
    template = (
    "Combine the chat history and follow up question into "
    "a standalone question. and Please answer in Korean. \n\n{context}"
    "Chat History: {chat_history}"
    "Follow up question: {question}"
    )
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    condense_question_prompt = PromptTemplate.from_template(template)
    # question_generator_chain = LLMChain(llm=llm, prompt=PROMPT)
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm=llm, retriever=index.vectorstore.as_retriever(), memory=memory, qaChainOptions={type: "stuff", prompt: QA_PROMPT})
    chat_response = conversation_with_retrieval({"question": input_text + '한국어로 답변해줘'})  
    print(chat_response)
    '''

    prompt_template = """
    Human: This is a friendly conversation between a human and an AI. 
    The AI is talkative and provides specific details from its context but limits it to 240 tokens.
    If the AI does not know the answer to a question, it truthfully says it does not know.

    Assistant: OK, got it, I'll be a talkative truthful AI assistant.

    Human: Here are a few documents in <documents> tags:
    <documents>
    {context}
    </documents>
    Based on the above documents, provide a detailed answer for, {question} 
    Answer "don't know" if not present in the document. 

    Assistant:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    condense_qa_template = """
    {chat_history}
    Human:
    Given the previous conversation and a follow-up question below, 
    rephrase the follow-up question to be a standalone question.

    Follow Up Question: {question}
    Standalone Question:

    Assistant:
    """
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    llm = get_llm()
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=index.vectorstore.as_retriever(), 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=False,
        memory=memory
    )
    chat_response = qa({"question": input_text + '한국어로 답변해줘'})  
    return chat_response['answer']