# Below code is required to fix sqlite error (unsupported version of sqlite3) while deploying
# app on streamlit community cloud
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
##########

from util import *
from dotenv import load_dotenv

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am an AI bot. How can I help you?")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# App Configuration
st.set_page_config(page_title="Ask the Web", page_icon=":robot_face:")
st.title("Ask The Web")
st.write(":blue[***Your Questions, Our Answers – From Any Website***]")
st.write("*Ask the Web* is a cutting-edge application designed to transform how you interact with websites. "
         "Simply input a website URL, and it will leverage advanced artificial intelligence to answer any question "
         "you have related to that website. ")

st.session_state.api_key, is_active = sidebar_api_key_configuration()
st.sidebar.divider()
st.sidebar.subheader("About")
configure_about_sidebar()

st.subheader('Enter Website URL')
website_url = st.text_input("Website URL", placeholder="Enter URL", label_visibility="collapsed",
                            disabled=not is_active)
process = st.button("Process", type="primary", disabled=not website_url)

if process:
    with st.spinner('Processing ...'):
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

st.divider()

# User Input
user_query = st.chat_input("Type your message here...", disabled=not website_url)
if user_query is not None and user_query != "":
    with st.spinner('Processing ...'):
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
