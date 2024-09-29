import json
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
host = os.getenv('OPENSEARCH_HOST')
user = os.getenv('OPENSEARCH_USER')
password = os.getenv('OPENSEARCH_PASSWORD')
region = os.getenv('OPENSEARCH_REGION')
index_name = os.getenv('OPENSEARCH_INDEX')

# Initialize OpenSearch client
os_client = OpenSearch(
    hosts=[{'host': host.replace("https://", ""), 'port': 443}],
    http_auth=(user, password),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Define index mapping
mapping = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512
        }
    },
    "mappings": {
        "properties": {
            "code": {"type": "keyword"},
            "title": {"type": "text"},
            "synopsis": {"type": "text"},
            "topics": {"type": "keyword"},
            "aws_services": {"type": "keyword"},
            "target_audience": {"type": "keyword"},
            "session_format": {"type": "keyword"},
            "title_embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "faiss",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            "synopsis_embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "faiss",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            }
        }
    }
}

def init_opensearch_index(os_client, index_name, mapping):
    if os_client.indices.exists(index=index_name):
        os_client.indices.delete(index=index_name)
    os_client.indices.create(index=index_name, body=mapping)

def download_document_with_emb():
    url = "https://d1jp7kj5nqor8j.cloudfront.net/session_info_with_emb.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def main():
    # Initialize OpenSearch index
    init_opensearch_index(os_client, index_name, mapping)

    # Download document_with_emb
    print("Downloading document_with_emb...")
    document_with_emb = download_document_with_emb()
    print("Download complete.")

    # Prepare bulk data
    bulk_data = []
    for doc in document_with_emb:
        bulk_data.append({"index": {"_index": index_name, "_id": doc['code']}})
        bulk_data.append(doc)

    # Index data
    if bulk_data:
        print("Indexing documents...")
        response = os_client.bulk(body=bulk_data)
        successful = sum(1 for item in response['items'] if item['index']['status'] in (200, 201))
        failed = len(response['items']) - successful

        print(f"Indexed {successful} documents successfully.")
        print(f"Failed to index {failed} documents.")
    else:
        print("No data to index.")

if __name__ == "__main__":
    main()