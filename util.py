import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Function for API configuration at sidebar
def sidebar_api_key_configuration():
    st.sidebar.subheader("API Keys")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password",
                                    help='Get API Key from: https://platform.openai.com/api-keys')
    if api_key == '':
        st.sidebar.warning('Enter the API Key')
        is_active = False
    elif api_key.startswith('sk-') and (len(api_key) == 51):
        st.sidebar.success('Lets Proceed!', icon='️👉')
        is_active = True
    else:
        st.sidebar.warning('Please enter the correct API Key!', icon='⚠️')
        is_active = False
    return api_key, is_active


def get_context_retriever_chain(vstore):
    llm = ChatOpenAI(openai_api_key=st.session_state.api_key)
    retriever = vstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query "
                 "to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(openai_api_key=st.session_state.api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the following context:{context}. If user question is out of "
                   "context, do not make up the answer and respond with I don't know the answer or similar."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(document)

    # Create vectorstore from chunks
    v_store = Chroma.from_documents(doc_chunks, OpenAIEmbeddings(openai_api_key=st.session_state.api_key))
    return v_store


def get_response(user_input):
    # Create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })
    return response['answer']
