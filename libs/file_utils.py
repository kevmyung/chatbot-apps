import os
import streamlit as st
import json
from PIL import Image


class CustomUploadedFile:
    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data
        self.file_id = name 

    def read(self, *args):
        return self.data.read(*args)

    def seek(self, *args):
        return self.data.seek(*args)

    def readlines(self):
        return self.data.readlines()
    
    def readline(self, *args):
        return self.data.readline(*args)
    
    def tell(self):
        return self.data.tell()


def process_uploaded_files(uploaded_files, existing_file_names):
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0
    content_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in existing_file_names:
            if uploaded_file.type.startswith('image/'):
                img = Image.open(uploaded_file)
                with cols[i]:
                    st.image(img, caption="", width=75)
                    i = (i + 1) % num_cols
                content_files.append({"name": uploaded_file.name, "type": "image", "content": None})
            elif uploaded_file.type == 'application/pdf':
                st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")
                content_files.append({"name": uploaded_file.name, "type": "pdf", "content": None})
            elif uploaded_file.type in ['text/plain', 'text/csv', 'text/x-python-script']:
                st.write(f"📄 Uploaded file: {uploaded_file.name}")
                content = uploaded_file.read().decode("utf-8")
                content_files.append({"name": uploaded_file.name, "type": "text", "content": content})
            else:
                st.write(f"📎 Uploaded file: {uploaded_file.name}")
                content_files.append({"name": uploaded_file.name, "type": "other", "content": None})

    return content_files


def handle_file_uploads(uploaded_files):
    for file in uploaded_files:
        if file not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(file)

    current_file_names = [file.name for file in uploaded_files]
    for message in st.session_state.messages:
        if "images" in message and message["images"]:
            message["images"] = [img for img in message["images"] if img in current_file_names]
        if "documents" in message and message["documents"]:
            message["documents"] = [doc for doc in message["documents"] if doc in current_file_names]


def process_user_input(prompt, uploaded_files=[], context=""):
    message_images_list = [
        image for message in st.session_state.messages if message["role"] == "user" and "images" in message and message["images"] for image in message["images"]
    ]
    message_documents_list = [
        doc for message in st.session_state.messages if message["role"] == "user" and "documents" in message and message["documents"] for doc in message["documents"]
    ]

    if uploaded_files and (len(message_images_list) < len(uploaded_files) or len(message_documents_list) < len(uploaded_files)):
        with st.chat_message("user"):
            content_files = process_uploaded_files(uploaded_files, message_images_list)
            if prompt:
                context_text, context_image, context_documents = "", [], []
                for content_file in content_files:
                    if content_file['type'] == 'text':
                        context_text += content_file['content'] + "\n\n"
                    elif content_file['type'] == 'image':
                        context_image.append(content_file['name'])
                    elif content_file['type'] in ['pdf', 'txt', 'csv']:
                        context_documents.append(content_file['name'])

                prompt_new = f"Here is some context for you: \n<context>\n{context_text}</context>\n\n{prompt}" if context_text else prompt
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt_new,
                    "images": context_image,
                    "documents": context_documents,
                    "user_prompt_only": prompt  
                })
                st.markdown(prompt)

    elif prompt:
        with st.chat_message("user"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(prompt)


def sample_query_indexing(os_client, lang_config):
    rag_query_file = st.text_input(lang_config['rag_query_file'], value="./db_metadata/chinook_example_queries.jsonl")
    if not os.path.exists(rag_query_file):
        st.warning(lang_config['file_not_found'])
        return

    if st.sidebar.button(lang_config['process_file'], key='query_file_process'):
        with st.spinner("Now processing..."):
            os_client.delete_index()
            os_client.create_index() 

            with open(rag_query_file, 'r') as file:
                bulk_data = file.read()

            response = os_client.conn.bulk(body=bulk_data)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")


def schema_desc_indexing(os_client, lang_config):
    schema_file = st.text_input(lang_config['schema_file'], value="./db_metadata/chinook_detailed_schema.json")
    if not os.path.exists(schema_file):
        st.warning(lang_config['file_not_found'])
        return

    if st.sidebar.button(lang_config['process_file'], key='schema_file_process'):
        with st.spinner("Now processing..."):
            os_client.delete_index()
            os_client.create_index() 

            with open(schema_file, 'r', encoding='utf-8') as file:
                schema_data = json.load(file)

            bulk_data = []
            for table in schema_data:
                for table_name, table_info in table.items():
                    table_doc = {
                        "table_name": table_name,
                        "table_desc": table_info["table_desc"],
                        "columns": [{"col_name": col["col"], "col_desc": col["col_desc"]} for col in table_info["cols"]],
                        "table_summary": table_info["table_summary"],
                        "table_summary_v": table_info["table_summary_v"]
                    }
                    bulk_data.append({"index": {"_index": os_client.index_name, "_id": table_name}})
                    bulk_data.append(table_doc)
            
            bulk_data_str = '\n'.join(json.dumps(item) for item in bulk_data) + '\n'

            response = os_client.conn.bulk(body=bulk_data_str)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")