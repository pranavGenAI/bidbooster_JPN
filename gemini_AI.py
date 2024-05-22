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

st.set_page_config(page_title="ビッドブースター", layout="wide")

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

        .animated-gradient-text {
            font-family: "Graphik Semibold";
            font-size: 42px;
            background: linear-gradient(45deg, rgb(245, 58, 126) 30%, rgb(200, 1, 200) 55%, rgb(197, 45, 243) 20%);
            background-size: 300% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientAnimation 10s ease-in-out infinite;
        }
    </style>
    <p class="animated-gradient-text">
        ビッドブースター: 入札プロセスを簡素化します!
    </p>
""", unsafe_allow_html=True)

# This is the first API key input; no need to repeat it in the main function.
#api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

api_key = st.secrets['GEMINI_API_KEY']
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
    与えられたコンテクストから、できるだけ詳しく日本語で質問に答えてください。答えの表現を変え、最も丁寧な言い方で始めてください。回答の要約を述べた後、詳細な回答を述べてください。丁寧で素敵な文章で始めてください。あいさつを使っても構いません。間違った回答をしないでください。コンテクストを使わず答えを出している場合、その答えがコンテクストに基づいていないことをユーザーに知らせてください。\n\n.そして、あなたの答えをより良い方法でフォーマットすることを忘れないでください。あなたの回答の最後には、その回答がコンテキストからのものであることを伝える免責事項を書いてください。丁寧な態度で、アシスタントの挨拶から回答を始めることを忘れないでください。
    コンテキスト:\n{context}?\n
    質問: \n{question}\n. 理解しやすい言葉で回答を書き、より良い方法でフォーマットする前に、コンテキストの文言を要約して変更していることを確認してください。質問の最後に、答えはコンテクストに基づくものであり、正確さは出典から確認する必要があるという免責事項を可能な限り明確に書いてください。Your answer should not include the user question.
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
    st.write("ビッドブースター: ", response["output_text"])

def main():
    st.header("質問してください...")

    user_question = st.text_input("RFPファイルから質問する", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        if st.button("Ask Question"):
            user_input(user_question, api_key)

    with st.sidebar:
        st.image("https://www.vgen.it/wp-content/uploads/2021/04/logo-accenture-ludo.png", width=150)
        st.markdown("")
        st.markdown("")
        
        st.title("ビッドブースター 💬")
        pdf_docs = st.file_uploader("RFP ファイルをアップロードし、「送信して処理」ボタンをクリックします。", accept_multiple_files=True, key="pdf_uploader")
        if st.button("送信して処理する", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("処理..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("終わり")

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
    main()
