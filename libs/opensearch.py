import os
import yaml
from copy import deepcopy
from typing import List, Optional, Dict
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.retrievers.self_query.base import SelfQueryRetriever


class OpenSearchClient:
    def __init__(self, llm, emb):
        config = self.load_opensearch_config()
        self.index_name = 'sample_index'
        self.llm = llm
        self.emb = emb
        self.config = config
        self.endpoint = config['opensearch-auth']['domain_endpoint']
        self.http_auth = (config['opensearch-auth']['user_id'], config['opensearch-auth']['user_password'])
        self.mapping = {"settings": config['settings'], "mappings": config['mappings']}
        self.conn = OpenSearch(
            hosts=[{'host': self.endpoint.replace("https://", ""), 'port': 443}],
            http_auth=self.http_auth, 
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        ) 
        self.vector_store = OpenSearchVectorSearch(
            index_name=self.index_name,
            opensearch_url=self.endpoint,
            embedding_function=self.emb,
            http_auth=self.http_auth,
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
        self.conn.indices.delete(self.index_name)


class OpenSearchRetriever(BaseRetriever):
    os_client: OpenSearchClient
    k: int = 5
    verbose: bool = True
    filter: List[dict] = []

    def __init__(self, os_client: OpenSearchClient):
        super().__init__(os_client=os_client)
        self.os_client = os_client
    
    def _get_relevant_documents(self, *, query: str, ensemble: List) -> List[Document]: 
        os_client = self.os_client
        search_result = retriever_utils.search_hybrid(
            query = query,
            k = self.k,
            filter = self.filter,
            index_name = os_client.index_name,
            os_conn = os_client.conn,
            llm = os_client.llm,
            emb = os_client.emb,
            ensemble_weights = ensemble
        )

        return search_result


class retriever_utils():
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=512,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    token_limit = 300

    @classmethod
    def search_hybrid(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "emb" in kwargs, "Check your emb"
        assert "index_name" in kwargs, "Check your index_name"
        assert "os_conn" in kwargs, "Check your OpenSearch Connection"
        
        verbose = kwargs.get("verbose", False)
        search_filter = deepcopy(kwargs.get("filter", []))
        ensemble_weights = kwargs.get("ensemble_weights")

        def search_sync():
            similar_docs_semantic = cls.get_semantic_similar_docs(
                index_name=kwargs["index_name"],
                os_conn=kwargs["os_conn"],
                emb=kwargs["emb"],
                query=kwargs["query"],
                k=kwargs.get("k", 5),
                boolean_filter=search_filter,
                hybrid=True
            )
            
            similar_docs_lexical = cls.get_lexical_similar_docs(
                index_name=kwargs["index_name"],
                os_conn=kwargs["os_conn"],
                query=kwargs["query"],
                k=kwargs.get("k", 5),
                minimum_should_match=kwargs.get("minimum_should_match", 0),
                filter=search_filter,
                hybrid=True
            )

            return similar_docs_semantic, similar_docs_lexical

        similar_docs_semantic, similar_docs_lexical = search_sync()
        print("semantic_docs:", similar_docs_semantic)
        print("lexical_docs:", similar_docs_lexical)
        print(kwargs.get("ensemble_weights", [.51, .49]))

        similar_docs = cls.get_ensemble_results(
            doc_lists=[similar_docs_semantic, similar_docs_lexical],
            weights=kwargs.get("ensemble_weights", [.51, .49]),
            c=60,
            k=kwargs.get("k", 5),
        )

        similar_docs = list(map(lambda x:x[0], similar_docs))
        
        return similar_docs

    @classmethod 
    def get_semantic_similar_docs(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "k" in kwargs, "Check your k"
        assert "os_conn" in kwargs, "Check your OpenSearch Connection"
        assert "index_name" in kwargs, "Check your index_name"

        def normalize_search_results(search_results):

            hits = (search_results["hits"]["hits"])
            max_score = float(search_results["hits"]["max_score"])
            for hit in hits:
                hit["_score"] = float(hit["_score"]) / max_score
            search_results["hits"]["max_score"] = hits[0]["_score"]
            search_results["hits"]["hits"] = hits
            return search_results

        query = get_opensearch_query(
            query=kwargs["query"],
            filter=kwargs.get("boolean_filter", []),
            search_type="semantic", 
            vector_field="vector_field", 
            vector=kwargs["emb"].embed_query(kwargs["query"]),
            k=kwargs["k"]
        )
        query["size"] = kwargs["k"]
        #query["min_score"] = 0.3

        search_results = lookup_opensearch_document(
            os_conn=kwargs["os_conn"],
            query=query,
            index_name=kwargs["index_name"]
        )
        results = []
        if search_results["hits"]["hits"]:
            search_results = normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:

                metadata = res["_source"]["metadata"]
                metadata["id"] = res["_id"]

                doc = Document(
                    page_content=res["_source"]["text"],
                    metadata=metadata
                )
                if kwargs.get("hybrid", False):
                    results.append((doc, res["_score"]))
                else:
                    results.append((doc))

        return results

    @classmethod
    def get_lexical_similar_docs(cls, **kwargs):

        assert "query" in kwargs, "Check your query"
        assert "k" in kwargs, "Check your k"
        assert "os_conn" in kwargs, "Check your OpenSearch Connection"
        assert "index_name" in kwargs, "Check your index_name"

        def normalize_search_results(search_results):

            hits = (search_results["hits"]["hits"])
            max_score = float(search_results["hits"]["max_score"])
            for hit in hits:
                hit["_score"] = float(hit["_score"]) / max_score
            search_results["hits"]["max_score"] = hits[0]["_score"]
            search_results["hits"]["hits"] = hits
            return search_results

        query = get_opensearch_query(query=kwargs["query"], filter=kwargs["filter"])
        query["size"] = kwargs["k"]

        search_results = lookup_opensearch_document(
            os_conn=kwargs["os_conn"],
            query=query,
            index_name=kwargs["index_name"]
        )

        results = []
        if search_results["hits"]["hits"]:
            search_results = normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:

                metadata = res["_source"]["metadata"]
                metadata["id"] = res["_id"]

                doc = Document(
                    page_content=res["_source"]["text"],
                    metadata=metadata
                )
                if kwargs.get("hybrid", False):
                    results.append((doc, res["_score"]))
                else:
                    results.append((doc))

        return results


    @classmethod
    def get_ensemble_results(cls, doc_lists: List[List[Document]], weights, c=60, k=5) -> List[Document]:
        all_documents = set()

        for doc_list in doc_lists:
            for (doc, _) in doc_list:
                all_documents.add(doc.page_content)

        hybrid_score_dic = {doc: 0.0 for doc in all_documents}    
        for doc_list, weight in zip(doc_lists, weights):
            for rank, (doc, score) in enumerate(doc_list, start=1):
                score *= weight
                hybrid_score_dic[doc.page_content] += score

        sorted_documents = sorted(hybrid_score_dic.items(), key=lambda x: x[1], reverse=True)

        page_content_to_doc_map = {doc.page_content: doc for doc_list in doc_lists for (doc, orig_score) in doc_list}

        sorted_docs = [(page_content_to_doc_map[page_content], hybrid_score) for (page_content, hybrid_score) in sorted_documents]

        return sorted_docs[:k]


def get_opensearch_retriever(os_client: OpenSearchClient):
    if "retriever" not in st.session_state:
        retriever = OpenSearchRetriever(os_client)
        st.session_state['retriever'] = retriever
    else:
        retriever = st.session_state['retriever']
    return retriever


def filter_by_min_score(results, min_score):
    filtered_results = []
    for result in results['hits']['hits']:
        if result['_score'] >= min_score:
            filtered_results.append(result)
    return filtered_results

def get_opensearch_query(query: str, filter: List[dict], search_type: str = "lexical", 
vector_field: Optional[str] = None, vector: Optional[List[float]] = None, k: int = 5) -> dict:

    if search_type == "lexical":
        query_template = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "operator": "or",
                                }
                            }
                        },
                    ],
                    "filter": filter
                }
            }
        }

    elif search_type == "semantic":
        if not vector_field or not vector:
            raise ValueError("vector_field and vector must be provided for semantic search")
        query_template = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                vector_field: {
                                    "vector": vector,
                                    "k": k,
                                }
                            }
                        },
                    ],
                    "filter": filter
                }
            }
        }

    return query_template


def lookup_opensearch_document(os_conn, query, index_name):
    response = os_conn.search(
        body=query,
        index=index_name
    )
    return response