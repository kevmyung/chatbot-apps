import os
import pickle
import requests
from typing import List, Union, Dict
from py2neo import Graph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

url = "https://d1jp7kj5nqor8j.cloudfront.net/bedrock_manual.pkl"
filename = 'samples/bedrock_manual.pkl'

class Node:
    def __init__(self, id: Union[str, int], type: str = "Node", properties: Dict = None):
        self.id = id
        self.type = type
        self.properties = properties or {}

class Relationship:
    def __init__(self, source: Node, target: Node, type: str, properties: Dict = None):
        self.source = source
        self.target = target
        self.type = type
        self.properties = properties or {}

class GraphDocument:
    def __init__(self, nodes: List[Node], relationships: List[Relationship], source: Dict = None):
        self.nodes = nodes
        self.relationships = relationships
        self.source = source or {}

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully: {filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def init_graph(graph, vector_search_enabled: bool):
    graph.run("MATCH (n) DETACH DELETE n")

    if vector_search_enabled:
        index_name = "content_embedding_index"
        query = """
            CREATE VECTOR INDEX $index_name IF NOT EXISTS
            FOR (n:Content) ON (n.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """
        params = {"index_name": index_name}
        graph.run(query, params)
    graph.run("""
        CREATE FULLTEXT INDEX Search_Content_by_FullText IF NOT EXISTS
        FOR (n:Content) 
        ON EACH [n.text] 
        OPTIONS { indexConfig: { `fulltext.analyzer`: "english"}}
        """)

download_file(url, filename)

with open(filename, 'rb') as f:
    loaded_graph_docs = pickle.load(f)

graph = Graph(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))

vector_search_enabled = True
index_name = "content_embedding_index"

init_graph(graph, vector_search_enabled)

def add_graph_documents(graph, graph_documents: List[GraphDocument], baseEntityLabel: bool = False):
    if baseEntityLabel:
        graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:`__Entity__`) REQUIRE b.id IS UNIQUE;")
    
    for document in graph_documents:
        for node in document.nodes:
            node_labels = [node.type]
            if baseEntityLabel:
                node_labels.append("__Entity__")
            node_properties = node.properties.copy()
            node_properties['id'] = node.id
            
            labels = ":".join(f"`{label}`" for label in node_labels)
            merge_node_query = f"""
                MERGE (n:{labels} {{id: $id}})
                SET n += $properties
            """
            graph.run(merge_node_query, id=node.id, properties=node_properties)
        
        for rel in document.relationships:
            source = rel.source
            target = rel.target
            
            source_labels = [source.type]
            if baseEntityLabel:
                source_labels.append("__Entity__")
            target_labels = [target.type]
            if baseEntityLabel:
                target_labels.append("__Entity__")
            
            source_label_str = ":".join(f"`{label}`" for label in source_labels)
            target_label_str = ":".join(f"`{label}`" for label in target_labels)
            rel_type = rel.type.replace(" ", "_").upper()
            rel_properties = rel.properties.copy()
            
            merge_rel_query = f"""
                MATCH (a:{source_label_str} {{id: $source_id}})
                MATCH (b:{target_label_str} {{id: $target_id}})
                MERGE (a)-[r:`{rel_type}`]->(b)
                SET r += $properties
            """
            graph.run(merge_rel_query, source_id=source.id, target_id=target.id, properties=rel_properties)

add_graph_documents(graph, loaded_graph_docs, baseEntityLabel=True)

print("Process completed successfully.")