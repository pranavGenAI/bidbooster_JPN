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

st.set_page_config(page_title="BidBooster", layout="wide")

st.markdown("""
## ビッドブースター 🤗💬: RFP 関連の質問に答えます.

### 使い方？

チャットボットと対話するには、次の簡単な手順に従ってください。
1. **RFP ドキュメントをアップロードし、[送信して処理] をクリックします** (注意: 基本 LLM モデルは LCBO ESG RFP ドキュメントに基づいて微調整されています。結果は他のドキュメントでは異なる場合があります)。
2. **質問する:** 文書が処理されたら、正確な回答を得るためにその内容に関連する質問をしてください。
3. より良い結果を得るために、**プロンプトが明確かつ完全である**ことを確認し、コンテキストを含めてください。
""")

st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjl2dGNiYThobHplMG81aGNqMjdsbWwwYWJmbTBncGp6dHFtZTFzMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/CGP9713UVzQ0BQPhSf/giphy.gif", width=50)

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
    提供された文脈から可能な限り詳細に、丁寧な方法で日本語で質問に答えてください。答えがそうでない場合は、必ずすべての詳細を要約した形式で提供してください。
    コンテキストが提供された場合は、「そのコンテキストでは答えがありません」とだけ言うだけで、間違った答えを提供しないでください。そして、回答をより適切な形式にすることを忘れないでください。
    コンテキストをコピーして貼り付けないでください。より適切な方法で要約してから、答えを提供してください。
    
    コンテクスト:\n {context}?\n
    質問: \n{question}\n .日本語で要約した回答を提供し、より適切な形式で作成してください。必要に応じて箇条書きを追加します。 RFP コンテキストをコピーして貼り付けるのではなく、要約してください。

    Answer:
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

    user_question = st.text_input("PDFファイルから質問する", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("ビッドブースター 🤗💬")
        pdf_docs = st.file_uploader("PDF ファイルをアップロードし、「送信して処理」ボタンをクリックします。", accept_multiple_files=True, key="pdf_uploader")
        if st.button("送信して処理する", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("処理..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("終わり")

        st.image("https://media.tenor.com/s1Y9XfdN08EAAAAi/bot.gif", width=200)


if __name__ == "__main__":
    main()
