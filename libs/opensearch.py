import os
import json
import yaml
import boto3
from copy import deepcopy
from typing import List, Optional, Dict, Tuple, Any
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection
from libs.file_utils import sample_query_indexing, schema_desc_indexing

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class Retriever:
    def get_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError


class OpenSearchClient:
    def __init__(self, index_name, mapping_name, vector, text, output):
        config = self.load_opensearch_config()
        self.index_name = index_name
        self.config = config
        self.endpoint = config['opensearch-auth']['domain_endpoint']
        self.http_auth = (config['opensearch-auth']['user_id'], config['opensearch-auth']['user_password'])
        self.vector = vector
        self.text = text
        self.output = output
        self.mapping = {"settings": config['settings'], "mappings": config[mapping_name]}
        self.conn = OpenSearch(
            hosts=[{'host': self.endpoint.replace("https://", ""), 'port': 443}],
            http_auth=self.http_auth, 
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        ) 

    def load_opensearch_config(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(file_dir, "opensearch.yml")

        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def is_index_present(self):
        return self.conn.indices.exists(self.index_name)
    
    def create_index(self):
        self.conn.indices.create(self.index_name, body=self.mapping)

    def delete_index(self):
        if self.is_index_present():
            self.conn.indices.delete(self.index_name)


class OpenSearchHybridRetriever:
    def __init__(self, os_client, k: int = 5):
        self.os_client = os_client
        self.k = k
        self.filter = []

    def get_relevant_documents(self, query: str, ensemble: List[float]) -> List[Document]:
        try:
            search_result = retriever_utils.search_hybrid(
                query=query,
                k=self.k,
                filter=self.filter,
                index_name=self.os_client.index_name,
                os_conn=self.os_client.conn,
                ensemble_weights=ensemble,
                vector_field=self.os_client.vector,
                text_field=self.os_client.text,
                output_field=self.os_client.output
            )
            return search_result
        except Exception as e:
            st.error(f"Error in retrieving documents: {str(e)}")
            st.warning("Unable to retrieve relevant documents. Please check your OpenSearch configuration.")
            return []

class retriever_utils():

    @classmethod 
    def normalize_search_results(cls, search_results):
        hits = (search_results["hits"]["hits"])
        max_score = float(search_results["hits"]["max_score"])
        for hit in hits:
            hit["_score"] = float(hit["_score"]) / max_score
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results
    
    @classmethod 
    def search_semantic(cls, **kwargs):
        boto3_client = boto3.client("bedrock-runtime", region_name=st.session_state.bedrock_region)
        model_id = "amazon.titan-embed-text-v2:0"
        request = json.dumps({"inputText": kwargs["query"]})
        response = boto3_client.invoke_model(modelId=model_id, body=request)
        embeddings = json.loads(response["body"].read())["embedding"]

        semantic_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                kwargs["vector_field"]: {
                                    "vector": embeddings,
                                    "k": kwargs["k"],
                                }
                            }
                        },
                    ],
                    "filter": kwargs.get("boolean_filter", []),
                }
            },
            "size": kwargs["k"],
            #"min_score": 0.3
        }
        # get semantic search results
        search_results = lookup_opensearch_document(
            index_name=kwargs["index_name"],
            os_conn=kwargs["os_conn"],
            query=semantic_query,
        )

        results = []
        if search_results.get("hits", {}).get("hits", []):
            # normalize the scores
            search_results = cls.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:
                if "metadata" in res["_source"]:
                    metadata = res["_source"]["metadata"]
                else:
                    metadata = {}
                metadata["id"] = res["_id"]

                # extract the text contents
                page_content = json.dumps({field: res["_source"].get(field, "") for field in kwargs["output_field"]})
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                results.append((doc, res["_score"]))
        return results

    @classmethod
    def search_lexical(cls, **kwargs):
        lexical_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                kwargs["text_field"]: {
                                    "query": kwargs["query"],
                                    "operator": "or",
                                }
                            }
                        },
                    ],
                    "filter": kwargs["filter"]
                }
            },
            "size": kwargs["k"] 
        }

        # get lexical search results
        search_results = lookup_opensearch_document(
            index_name=kwargs["index_name"],
            os_conn=kwargs["os_conn"],
            query=lexical_query,
        )

        results = []
        if search_results.get("hits", {}).get("hits", []):
            # normalize the scores
            search_results = cls.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:
                if "metadata" in res["_source"]:
                    metadata = res["_source"]["metadata"]
                else:
                    metadata = {}
                metadata["id"] = res["_id"]

                # extract the text contents
                page_content = json.dumps({field: res["_source"].get(field, "") for field in kwargs["output_field"]})
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                results.append((doc, res["_score"]))
        return results

    @classmethod
    def search_hybrid(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "index_name" in kwargs, "Check your index_name"
        assert "os_conn" in kwargs, "Check your OpenSearch Connection"

        search_filter = deepcopy(kwargs.get("filter", []))
        similar_docs_semantic = cls.search_semantic(
                index_name=kwargs["index_name"],
                os_conn=kwargs["os_conn"],
                query=kwargs["query"],
                k=kwargs.get("k", 5),
                vector_field=kwargs["vector_field"],
                output_field=kwargs["output_field"],
                boolean_filter=search_filter,
            )
        # print("semantic_docs:", similar_docs_semantic)

        similar_docs_lexical = cls.search_lexical(
                index_name=kwargs["index_name"],
                os_conn=kwargs["os_conn"],
                query=kwargs["query"],
                k=kwargs.get("k", 5),
                text_field=kwargs["text_field"],
                output_field=kwargs["output_field"],
                minimum_should_match=kwargs.get("minimum_should_match", 1),
                filter=search_filter,
            )
        # print("lexical_docs:", similar_docs_lexical)

        similar_docs = retriever_utils.get_ensemble_results(
            doc_lists=[similar_docs_semantic, similar_docs_lexical],
            weights=kwargs.get("ensemble_weights", [.51, .49]),
            k=kwargs.get("k", 5),
        )
        
        return similar_docs        


    @classmethod
    def get_ensemble_results(cls, doc_lists: List[List[Tuple[Document, float]]], weights: List[float], k: int = 5) -> List[Document]:
        hybrid_score_dic: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Weight-based adjustment
        for doc_list, weight in zip(doc_lists, weights):
            for doc, score in doc_list:
                doc_id = doc.metadata.get("id", doc.page_content)
                if doc_id not in hybrid_score_dic:
                    hybrid_score_dic[doc_id] = 0.0
                hybrid_score_dic[doc_id] += score * weight
                doc_map[doc_id] = doc

        sorted_docs = sorted(hybrid_score_dic.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]


def get_opensearch_retriever(os_client):
    if "retriever" not in st.session_state:
        retriever = OpenSearchHybridRetriever(os_client, 5)
        st.session_state['retriever'] = retriever
    else:
        retriever = st.session_state['retriever']
    return retriever


def lookup_opensearch_document(index_name, os_conn, query):
    response = os_conn.search(
        index=index_name,
        body=query
    )
    return response


def initialize_os_client(client_params: Dict, indexing_function, lang_config: Dict):
    client = OpenSearchClient(**client_params)
    indexing_function(client, lang_config)
    
    return client

def init_opensearch(lang_config):
    try:
        with st.sidebar:
            sql_os_client = initialize_os_client(
                {
                    "index_name": 'example_queries',
                    "mapping_name": 'mappings-sql',
                    "vector": "input_v",
                    "text": "input",
                    "output": ["input", "query"]
                },
                sample_query_indexing,
                lang_config
            )

            schema_os_client = initialize_os_client(
                {
                    "index_name": 'schema_descriptions',
                    "mapping_name": 'mappings-detailed-schema',
                    "vector": "table_summary_v",
                    "text": "table_summary",
                    "output": ["table_name", "table_summary"]
                },
                schema_desc_indexing,
                lang_config
            )

        return sql_os_client, schema_os_client

    except ValueError as e:
        st.error(f"OpenSearch configuration error: {str(e)}")
        st.warning("Unable to initilize OpenSearch. Please check your OpenSearch configuration (`libs/opensearch.yml`).")
        return None, None