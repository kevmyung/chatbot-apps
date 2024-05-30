from typing import List, Union
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .models import ChatModel
from .opensearch import OpenSearchClient
import os
import tempfile
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import streamlit as st

FAISS_PATH = './vectorstore/db_faiss'
FAISS_ORIGIN = './vectorstore/pdf' 
INDEX_FILE = 'index.faiss'

def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], message_images_list: List[str], uploaded_file_ids: List[str]) -> List[Union[dict, str]]:
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0
    content_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file.file_id not in message_images_list:
            uploaded_file_ids.append(uploaded_file.file_id)
            try:
                # Try to open as an image
                img = Image.open(uploaded_file)
                with BytesIO() as output_buffer:
                    img.save(output_buffer, format=img.format)
                    content_image = base64.b64encode(output_buffer.getvalue()).decode("utf8")
                content_files.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": content_image,
                    },
                })
                with cols[i]:
                    st.image(img, caption="", width=75)
                    i += 1
                if i >= num_cols:
                    i = 0
            except UnidentifiedImageError:
                # If not an image, try to read as a text or pdf file
                if uploaded_file.type in ['text/plain', 'text/csv', 'text/x-python-script']:
                    # Ensure we're at the start of the file
                    uploaded_file.seek(0)
                    # Read file line by line
                    lines = uploaded_file.readlines()
                    text = ''.join(line.decode() for line in lines)
                    content_files.append({
                        "type": "text",
                        "text": text
                    })
                    if uploaded_file.type == 'text/x-python-script':
                        st.write(f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == 'application/pdf':
                    # Read pdf file
                    pdf_file = pdfplumber.open(uploaded_file)
                    page_text = ""
                    for page in pdf_file.pages:
                        page_text += page.extract_text()
                    content_files.append({
                        "type": "text",
                        "text": page_text
                    })
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")
                    pdf_file.close()

    return content_files


def pdf_to_images(pdf_filepath):
    output_folder = os.path.splitext(pdf_filepath)[0] + "_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        images = convert_from_path(pdf_filepath)
        for i, image in enumerate(images):
            image_filename = os.path.join(output_folder, f"page_{i}.png")
            print(f"Image saved in {image_filename}.")
            image.save(image_filename, "PNG")
    except Exception as e:
        print(f"Failed to convert PDF to images: {e}")


def faiss_preprocess_document(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], 
                              chat_model: ChatModel, 
                              upload_message: str) -> FAISS:
    if uploaded_files:
        docs = []
        if not os.path.exists(FAISS_ORIGIN):
            os.makedirs(FAISS_ORIGIN)
        for file in uploaded_files:
            pdf_path = os.path.join(FAISS_ORIGIN, file.name)
            with open(pdf_path, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())

            pdf_to_images(pdf_path)

        # chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # embed & store
        if st.session_state['vector_empty'] == False:
            vectordb = FAISS.load_local(folder_path=FAISS_PATH, embeddings=chat_model.emb, allow_dangerous_deserialization=True)
            vectordb.add_documents(documents=splits, embeddings=chat_model.emb)
            vectordb.save_local(FAISS_PATH)
        else:
            vectordb = FAISS.from_documents(documents=splits, embedding=chat_model.emb)
            vectordb.save_local(FAISS_PATH)
            st.session_state['vector_empty'] = False

        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
        with st.sidebar:
            st.success(upload_message)
    else:
        if os.path.exists(f"{FAISS_PATH}/{INDEX_FILE}"):
            vectordb = FAISS.load_local(folder_path=FAISS_PATH, embeddings=chat_model.emb, allow_dangerous_deserialization=True)
            retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
            st.session_state['vector_empty'] = False
        else:
            retriever = None
    return retriever


def sample_query_indexing(os_client, lang_config):
    rag_query_file = st.text_input(lang_config['rag_query_file'], value="libs/example_queries.json")
    if not os.path.exists(rag_query_file):
        st.warning(lang_config['file_not_found'])

    if st.sidebar.button(lang_config['process_query']):
        with st.spinner("Now processing..."):
            if os_client.is_index_present:
                os_client.delete_index()
            os_client.create_index() 

            with open(rag_query_file, 'r') as file:
                bulk_data = file.read()

            response = os_client.conn.bulk(body=bulk_data)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")
    

def opensearch_preprocess_document(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, 
                                   os_client: OpenSearchClient, 
                                   upload_message: str):
    if uploaded_file:
        if not os_client.is_index_present():
            os_client.create_index()
        
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # embed & store
        os_client.vector_store.add_documents(documents=splits)
        st.session_state['vector_empty'] = False
        
        with st.sidebar:
            st.success(upload_message)
    else:
        if os_client.is_index_present(): 
            st.session_state['vector_empty'] = False


def opensearch_reset_on_click() -> None:
    if "os_client" in st.session_state:
        os_client = st.session_state['os_client']
        if os_client.is_index_present():
            os_client.delete_index()
    st.session_state['vector_empty'] = True
    st.success(st.session_state['clean_kb_message'])


def reset_faiss_index() -> None:
    import shutil
    shutil.rmtree(FAISS_PATH, ignore_errors=True)
    shutil.rmtree(FAISS_ORIGIN, ignore_errors=True)
    st.session_state['vector_empty'] = True
    st.success(st.session_state['clean_kb_message'])

def faiss_reset_on_click() -> None:
    reset_faiss_index()
