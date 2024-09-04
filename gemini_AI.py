import streamlit as st
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores import FAISS
import hashlib
import json

st.set_page_config(page_title="Gebotsabfrage-Bot", layout="wide")
video_html = """
		<style>
		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		  filter: brightness(20%); /* Adjust the brightness to make the video darker */
		}
		
		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.2); /* Adjust the transparency as needed */
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}
		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://assets.mixkit.co/videos/4907/4907-720.mp4" type="video/mp4">
		  Your browser does not support HTML5 video.
		</video>
		"""

st.markdown(video_html, unsafe_allow_html=True)

st.markdown("""
    <style>
        @keyframes gradientAnimation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .animated-gradient-text_ {
            font-size: 42px;
            background: linear-gradient(45deg, rgb(245, 58, 126) 30%, rgb(200, 1, 200) 55%, rgb(197, 45, 243) 20%);
            background-size: 300% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientAnimation 10s ease-in-out infinite;
        }
        @keyframes animate_ {
            0%, 18%, 20%, 50.1%,60%, 65.1%, 80%,90.1%,92% {
                color: #0e3742;
                text-shadow: none;
                }
            18.1%, 20.1%, 30%,50%,60.1%,65%,80.1%,90%, 92.1%,100% {
                color: #fff;
                text-shadow: 0 0 10px rgb(197, 45, 243),
                             0 0 20px rgb(197, 45, 243);
                }
            }
        
        .animated-gradient-text_ {
                    font-size: 42px;
                    color: #FFF;
                    transition: color 0.5s, text-shadow 0.5s;
                }

        .animated-gradient-text_:hover {
                    animation: animate_ 5s linear infinite;
                }

        
    </style>
    <p class="animated-gradient-text_">
        Gebotsabfrage-Bot: Den Bietprozess vereinfachen!
    </p>
""", unsafe_allow_html=True)

# This is the first API key input; no need to repeat it in the main function.
#api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

api_key = st.secrets['GEMINI_API_KEY']



# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Define users and hashed passwords for simplicity
users = {
    "pranav.baviskar": hash_password("pranav123")
}


TOKEN_FILE = "./data/token_counts_jpn.json"


def read_token_counts():
    try:
        with open("./data/token_counts_jpn.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def write_token_counts(token_counts):
    with open("./data/token_counts_jpn.json", "w") as f:
        json.dump(token_counts, f)


def get_token_count(username):
    token_counts = read_token_counts()
    return token_counts.get(username, 1000)  # Default to 1000 tokens if not found

def update_token_count(username, count):
    token_counts = read_token_counts()
    token_counts[username] = count
    write_token_counts(token_counts)


def login():
    col1, col2, col3 = st.columns([1, 1, 1])  # Create three columns with equal width
    with col2:  # Center the input fields in the middle column
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Sign in"):
            hashed_password = hash_password(password)
            if username in users and users[username] == hashed_password:
                token_counts = read_token_counts()
                tokens_remaining = token_counts.get(username, 500)  # Default to 500 tokens if not found
                
                if tokens_remaining > 0:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.tokens_remaining = tokens_remaining
                    st.session_state.tokens_consumed = 0
                    st.success("Logged in successfully!")
                    st.experimental_rerun()  # Refresh to show logged-in state
                else:
                    st.error("No tokens remaining. Please contact support.")
            else:
                st.error("Invalid username or password")
    # Add the footer section
    col4, col5, col6, col7, col8, col9 = st.columns([1, 1, 1, 1, 1, 1])
    st.markdown("")
    st.markdown("")
    with col7:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.write("**Design & Developed by:**")
    with col9:
        st.image("https://i.ibb.co/YRH1647/Pranav-Baviskar.png", width=150)
    with col8:
        st.image("https://i.ibb.co/0ssNpmD/Ritwick-Das.png", width=150)

def logout():
    # Clear session state on logout
    st.session_state.logged_in = False
    del st.session_state.username
    del st.session_state.tokens_remaining
    del st.session_state.tokens_consumed
    st.success("Logged out successfully!")
    st.experimental_rerun()  # Refresh to show logged-out state


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
	Betrachte dich selbst als den "Bid Query Bot", der Antworten auf Benutzeranfragen so detailliert wie möglich basierend auf dem bereitgestellten Kontext gibt. Beginne mit dem höflichsten Ausdruck, fasse deine Antwort zusammen und gib dann eine detaillierte Antwort.
	
	Kontext: {context}
	
	Frage: {question}
	
	Am Ende der Antwort sollte klar angegeben werden, dass die Antwort auf dem Kontext basiert und die Genauigkeit anhand der Quelle überprüft werden sollte.
	"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    print("Prompt ***** --->", prompt)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    #new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Gebotsabfrage-Bot: ", response["output_text"])
    num_words = len(response["output_text"].split())

    # Deduct tokens based on number of words
    token_cost = num_words  # Each word in the response costs 1 token (adjust as needed)
    
    # Check if enough tokens are available
    if st.session_state.tokens_remaining > 0:
        # Proceed with displaying the response and deducting tokens
        st.write("Bid Query Bot: ", response["output_text"])
        st.session_state.tokens_consumed += token_cost  # Deduct tokens based on response length
        st.session_state.tokens_remaining -= token_cost

        # Update token count in JSON file
        token_counts = read_token_counts()
        token_counts[st.session_state.username] = st.session_state.tokens_remaining
        write_token_counts(token_counts)
    else:
        st.warning("You don't have enough tokens. Please contact your administrator.")

    # Display remaining tokens to the user
    st.sidebar.text(f"Tokens Remaining: {st.session_state.tokens_remaining}")



def main():
    st.header("Bitte stellen Sie eine Frage...")

    user_question = st.text_input("Fragen aus der RFP-Datei stellen", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        if st.button("Ask Question"):
            user_input(user_question, api_key)

    with st.sidebar:
        st.image("https://www.vgen.it/wp-content/uploads/2021/04/logo-accenture-ludo.png", width=150)
        st.markdown("")
        st.markdown("")
        st.markdown("""
            <style>
                @keyframes animate {
                    0%, 18%, 20%, 50.1%,60%, 65.1%, 80%,90.1%,92% {
                        color: #0e3742;
                        text-shadow: none;
                    }
                    18.1%, 20.1%, 30%,50%,60.1%,65%,80.1%,90%, 92.1%,100% {
                        color: #fff;
                        text-shadow: 0 0 10px #03bcf4,
                                    0 0 20px #03bcf4,
                                    0 0 40px #03bcf4,
                                    0 0 80px #03bcf4,
                                    0 0 160px #03bcf4;
                    }
                }

                .animated-gradient-text {
                    font-family: "Graphik Semibold";
                    font-size: 26px;
                    color: #FFF;
                    transition: color 0.5s, text-shadow 0.5s;
                }

                .animated-gradient-text:hover {
                    animation: animate 5s linear infinite;
                }

            </style>
            <p class = animated-gradient-text> Gebotsabfrage-Bot 💬 </p>    

        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader("Laden Sie die RFP-Datei hoch und klicken Sie auf die Schaltfläche 'Absenden und Verarbeiten'", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Absenden und Verarbeiten", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Verarbeitung..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Ende")

       # st.image("https://media.tenor.com/s1Y9XfdN08EAAAAi/bot.gif", width=150)


if __name__ == "__main__":
    
    st.markdown('''<style>
        .stApp > header {
        background-color: transparent;
    }
    .stApp {
        background: linear-gradient(45deg, #0a1621 20%, #0E1117 45%, #0E1117 55%, #3a5683 90%);
        animation: my_animation 20s ease infinite;
        background-size: 200% 200%;
        background-attachment: fixed;
    }
    @keyframes my_animation {
        0% {background-position: 0% 0%;}
        50% {background-position: 100% 100%;}
        100% {background-position: 0% 0%;}
    }
    [data-testid=stSidebar] {
        background: linear-gradient(360deg, #1a2631 95%, #161d29 10%);
    }
    div.stButton > button:first-child {
        background:linear-gradient(45deg, #c9024b 45%, #ba0158 55%, #cd006d 70%);
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background:linear-gradient(45deg, #ce026f 45%, #970e79 55%, #6c028d 70%);
        background-color:#ce1126;
    }
    div.stButton > button:active {
        position:relative;
        top:3px;
    }    

    </style>''', unsafe_allow_html=True)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "tokens_consumed" not in st.session_state:
        st.session_state.tokens_consumed = 0
    if "tokens_remaining" not in st.session_state:
        st.session_state.tokens_remaining = 0
    
    if st.session_state.logged_in:
        st.sidebar.write(f"Welcome, {st.session_state.username}")
        st.sidebar.write(f"Tokens remaining: {st.session_state.tokens_remaining}")
        if st.sidebar.button("Logout"):
            logout()
        main()
    else:
        login()
