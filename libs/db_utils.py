# Standard Library Imports
import csv
import os
import json
import uuid
import re
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated
from botocore.config import Config
import boto3
import pandas as pd
import ast
import pytz
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning

# LangChain and Related Imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_xml_agent
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field

# Custom Module Imports
from .prompts import (
    get_sql_prompt, 
    get_table_selection_prompt, 
    get_query_generation_prompt, 
    get_prompt_refinement_prompt, 
    get_sample_selection_prompt, 
    get_query_validation_prompt,
    get_global_tool_prompt
)
from .opensearch import OpenSearchHybridRetriever, OpenSearchClient

warnings.filterwarnings("ignore", r".*support Decimal objects natively, and SQLAlchemy", SAWarning)
logging.basicConfig(level=logging.ERROR, filename='error.log')

class DatabaseClient:
    def __init__(self, llm, config, sql_os_client, schema_os_client):
        self.llm = llm
        self.dialect = config['dialect']
        self.top_k = 5
        self.sql_os_client = sql_os_client
        self.schema_os_client = schema_os_client
        self.db = SQLDatabase.from_uri(config['uri'])
        self.sql_toolkit = self.initialize_sql_toolkit()
        sql_tools = self.sql_toolkit.get_tools()        

        prompt = get_sql_prompt()
        agent = create_xml_agent(
            llm=self.llm,
            tools=sql_tools,
            prompt=prompt
        )
        self.sql_executor = AgentExecutor(agent=agent, tools=sql_tools, max_iterations=10) 

    def initialize_sql_toolkit(self):
        if self.schema_os_client:
            return CustomSQLDatabaseToolkit(db=self.db, llm=self.llm, schema_os_client=self.schema_os_client)
        else:
            return SQLDatabaseToolkit(db=self.db, llm=self.llm)

    def get_sample_queries(self, prompt):
        if self.sql_os_client:
            sql_os_retriever = OpenSearchHybridRetriever(self.sql_os_client, 10)
            samples = sql_os_retriever.invoke(prompt, ensemble = [0.51, 0.49])
            return samples
        else:
            return ""

class CustomListSQLDatabaseTool(ListSQLDatabaseTool):
    """Tool for getting tables names."""
    name: str = "custom_sql_db_list_tables"
    description: str = "Input is an empty string, output is a JSON object with table names as keys and their descriptions as values."
    table_descriptions: dict = {}

    def __init__(self, db, schema_os_client, **kwargs):
        super().__init__(db=db, **kwargs)
        self.table_descriptions = self.load_table_descriptions(schema_os_client)

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a JSON object with table names as keys and their descriptions as values."""
        table_names = self.db.get_usable_table_names()
        response = {table_name: self.table_descriptions.get(table_name, "No description available") for table_name in table_names}
        return response

    def load_table_descriptions(self, schema_os_client):
        query = {
            "_source": ["table_name", "table_desc"],
            "size": 1000,
            "query": {
                "match_all": {}
            }
        }
        response = schema_os_client.conn.search(index=schema_os_client.index_name, body=query)
        table_descriptions = {}
        for hit in response['hits']['hits']:
            source = hit['_source']
            table_name = source['table_name']
            table_desc = source['table_desc']
            table_descriptions[table_name] = {
                'table_desc': table_desc
            }
        return table_descriptions

class CustomInfoSQLDatabaseTool(InfoSQLDatabaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "custom_sql_db_schema"
    description: str = "Get the detailed schema and sample rows for the specified SQL tables."
    schema_os_client: OpenSearchClient

    def __init__(self, schema_os_client, **kwargs):
        super().__init__(schema_os_client=schema_os_client, **kwargs)
        self.schema_os_client = schema_os_client

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a dictionary with the table structure:
        {"table":"table_name", "cols": {"col":"column description", ...}, "create_table_sql":"CREATE_TABLE ...", "sample_data": "sample row data"} """
        tables = [t.strip() for t in table_names.split(",")]
        sql_statements = {}
        sample_data = {}
        data = self.db.get_table_info_no_throw(tables).strip()
        statements = data.split("\n\n")

        import re
        for statement in statements:
            if "CREATE TABLE" in statement:
                table_match = re.search(r"CREATE TABLE `(\w+)`", statement)
                if table_match:
                    table_name = table_match.group(1)
                    sql_statements[table_name] = statement
            elif "rows from" in statement:
                table_name_match = re.search(r"rows from (\w+) table", statement)
                if table_name_match:
                    table_name = table_name_match.group(1)
                    sample_data[table_name] = statement.strip()

        table_details = {}
        for table in tables:
            table_desc = self.get_column_description(table, self.schema_os_client)
            table_details[table] = {
                "table": table,
                "cols": table_desc if table_desc else {},
                "create_table_sql": sql_statements.get(table, "Not available"),
                "sample_data": sample_data.get(table, "No sample data available")
            }
            
            if not table_details[table]["cols"]:
                print(f"No columns found for table {table}")
            
        return table_details

    def get_column_description(self, table_name, schema_os_client):
        query = {
            "query": {
                "match": {
                    "table_name": table_name
                }
            }
        }
        response = schema_os_client.conn.search(index=schema_os_client.index_name, body=query)

        if response['hits']['total']['value'] > 0:
            source = response['hits']['hits'][0]['_source']
            columns = source.get('columns', [])
            if columns:
                return {col['col_name']: col['col_desc'] for col in columns}
            else:
                return {}
        else:
            return {}


class CustomSQLDatabaseToolkit(BaseToolkit):
    """
    Toolkit for interacting with SQL databases, now using the custom list tool.
    """
    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    schema_os_client: OpenSearchClient

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db, llm, schema_os_client, **kwargs):
        super().__init__(db=db, llm=llm, schema_os_client=schema_os_client, **kwargs)
        self.schema_os_client = schema_os_client

    def get_tools(self) -> List[BaseTool]:
        list_sql_database_tool = CustomListSQLDatabaseTool(db=self.db, schema_os_client=self.schema_os_client)  
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "description about the DB schema and sample rows for those tables. "
            "Use this tool before generating a query. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! ")
        info_sql_database_tool = CustomInfoSQLDatabaseTool(
            db=self.db, schema_os_client=self.schema_os_client, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a result from the database. "
            "If the query is not correct, an error message will be returned. "
            "If an error is returned, rewrite the query, check the query, and try again. "
            "If you encounter an issue with Unknown column 'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields. "
            "Only one statement can be executed at a time, so if multiple queries need to be executed, use this tool repeatedly."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct"
            f"before executing a query with f{query_sql_database_tool.name}!."
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]

class DB_Tools:
    def __init__(self, tokens: dict, uri: str, dialect: str, model: str, region: str, sql_os_client: OpenSearchClient, schema_os_client: OpenSearchClient, language: str, prompt: str):
        self.tokens = tokens
        self.uri = uri
        self.dialect = dialect
        self.model = model
        self.region = region
        self.language = language
        self.prompt = prompt
        self.sql_os_client = sql_os_client
        self.schema_os_client = schema_os_client
        self.engine = create_engine(uri)
        self.db = SQLDatabase(self.engine)
        self.client = self.init_boto3_client(region)
        self.samples = self.collect_samples()
        self.display_samples()
        self.trial = 0
        self.failure_log = ""
        self.failed_query = ""

    def init_boto3_client(self, region: str):
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", region_name=region, config=retry_config)

    def update_tokens(self, res):
        self.tokens["total_input_tokens"] += res["usage"]["inputTokens"]
        self.tokens["total_output_tokens"] += res["usage"]["outputTokens"]

    def collect_samples(self):
        with st.spinner("Collecting Sample Queries..."):
            return self.get_sample_queries()

    def display_samples(self):
        with st.expander("Referenced Sample Queries (Click to expand)", expanded=False):
            self.print_sql_samples(self.samples)

    def print_sql_samples(self, selected_samples: List[str]) -> None:
        for page_content in selected_samples:
            try:
                page_content_dict = json.loads(page_content)
                for key, value in page_content_dict.items():
                    if key == 'query':
                        st.markdown(f"```\n{value}\n```")
                    else:
                        st.markdown(f"{value}")
                st.markdown('<div style="margin: 5px 0;"><hr style="border: none; border-top: 1px solid #ccc; margin: 0;" /></div>', unsafe_allow_html=True)
            except json.JSONDecodeError:
                st.text("Invalid page_content format")  

    def get_sample_queries(self):
        if not self.sql_os_client:
            return []
        
        sql_os_retriever = OpenSearchHybridRetriever(self.sql_os_client, 10)
        samples = sql_os_retriever.invoke(self.prompt, ensemble=[0.40, 0.60])
        page_contents = [doc.page_content for doc in samples if doc is not None]
        sample_inputs = [json.loads(content)['input'] for content in page_contents]

        sys_prompt, usr_prompt = get_sample_selection_prompt(sample_inputs, self.prompt)
        response = self.client.converse(modelId=self.model, messages=usr_prompt, system=sys_prompt)
        self.update_tokens(response)
        sample_ids = response['output']['message']['content'][0]['text']

        sample_ids_list = [int(id.strip()) for id in sample_ids.split(',') if id.strip()]
        selected_samples = [page_contents[id] for id in sample_ids_list] if sample_ids_list else []

        return selected_samples

    def get_table_summaries_by_similarities(self):
        if self.schema_os_client:
            sql_os_retriever = OpenSearchHybridRetriever(self.schema_os_client, 5)
            matched_tables = sql_os_retriever.invoke(self.prompt, ensemble = [0.40, 0.60])
            serializable_tables = []
            for document in matched_tables:
                table_data = json.loads(document.page_content)
                serializable_tables.append(table_data)
            
            return json.dumps(serializable_tables, ensure_ascii=False)
        else:
            return ""

    def get_table_summaries_all(self):
        query = {
            "_source": ["table_name", "table_summary"],
            "size": 1000,
            "query": {
                "match_all": {}
            }
        }
        response = self.schema_os_client.conn.search(index=self.schema_os_client.index_name, body=query)
        table_descriptions = {}
        for hit in response['hits']['hits']:
            source = hit['_source']
            table_name = source['table_name']
            table_desc = source['table_summary']
            table_descriptions[table_name] = table_desc
        return table_descriptions

    def get_column_description(self, table_name: str) -> Dict[str, str]:
        query = {
            "_source": ["columns.col_name", "columns.col_desc"],
            "query": {
                "match": {
                    "table_name": table_name
                }
            }
        }
        response = self.schema_os_client.conn.search(index=self.schema_os_client.index_name, body=query)
        if response['hits']['total']['value'] > 0:
            source = response['hits']['hits'][0]['_source']
            columns = source.get('columns', [])
            if columns:
                return {col['col_name']: col['col_desc'] for col in columns}
            else:
                return {}
        else:
            return {}

    def get_table_schemas(self, table_names: List[str]) -> Dict[str, Dict]:
        try:
            tables = [t.strip() for t in table_names]
            sql_statements = {}
            sample_data = {}
            data = self.db.get_table_info_no_throw(tables)
            
            if not data:
                logging.warning("No data returned from DB")
                return {}

            statements = data.split("\n\n")

            for statement in statements:
                if "CREATE TABLE" in statement:
                    table_match = re.search(r"CREATE TABLE `(\w+)`", statement)
                    if table_match:
                        table_name = table_match.group(1)
                        sql_statements[table_name] = statement
                elif "rows from" in statement:
                    table_name_match = re.search(r"rows from (\w+) table", statement)
                    if table_name_match:
                        table_name = table_name_match.group(1)
                        sample_data[table_name] = statement.strip()

            table_details = {}
            for table in tables:
                table_desc = self.get_column_description(table) if self.schema_os_client else {}
                table_details[table] = {
                    "table": table,
                    "cols": table_desc if table_desc else {},
                    "create_table_sql": sql_statements.get(table, "Not available"),
                    "sample_data": sample_data.get(table, "No sample data available")
                }
                
                if not table_details[table]["cols"]:
                    print(f"No columns found for table {table}")

            return table_details

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return {}
    
    def get_explain_query(self, original_query):
        explain_statements = {
            'mysql': "EXPLAIN {query}",
            'mariadb': "EXPLAIN {query}",
            'sqlite': "EXPLAIN QUERY PLAN {query}",
            'oracle': "EXPLAIN PLAN FOR\n{query}\n\nSELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);",
            'postgresql': "EXPLAIN ANALYZE {query}",
            'postgres': "EXPLAIN ANALYZE {query}",
            'redshift': "EXPLAIN ANALYZE {query}",
            'presto': "EXPLAIN ANALYZE {query}",
            'sqlserver': "SET STATISTICS PROFILE ON; {query} SET STATISTICS PROFILE OFF;",
            'bigquery': "BigQuery requires using the API to get query explanation."
        }
        return explain_statements.get(self.dialect.lower(), f"Unsupported dialect: {self.dialect}. Please provide the EXPLAIN syntax manually.").format(query=original_query)

    def parse_json_format(self, json_string):
        json_string = re.sub(r'```json|```|</?response_format>|\n\s*', ' ', json_string)
        json_string = json_string.strip()
        match = re.search(r'({.*})', json_string)
        if match:
            json_string = match.group(1)
        else:
            return "No JSON object found in the string."

        try:
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print("Original output: ", json_string)
            return f"JSON Parsing Error: {e}"
        return parsed_json

    def save_to_csv(self, data, output_columns: List[str], query: str):
        try:
            data = ast.literal_eval(data)
        except (ValueError, SyntaxError) as ve:
            logging.error(f"Data conversion error: {ve}")
            return {"failure_log": "There was an error processing the query results. Please check the query syntax and output columns."}
        
        if data:
            df = pd.DataFrame(data, columns=output_columns)
            df = df.where(pd.notnull(df), None)
            
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            random_id = str(uuid.uuid4())
            folder_path = "./result_files"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            csv_file = f"{folder_path}/query_result_{current_time}_{random_id}.csv"
            query_file = f"{folder_path}/query_{current_time}_{random_id}.sql"

            df.to_csv(csv_file, index=False)
            with open(query_file, 'w') as file:
                file.write(query)
            return csv_file, query_file, df

    def prompt_refinement(self, input: str):
        query = {
            "_source": ["table_name", "table_desc"],
            "query": {
                "match_all": {}
            }
        }
        response = self.schema_os_client.conn.search(index=self.schema_os_client.index_name, body=query)
        table_descriptions = {}
        for hit in response['hits']['hits']:
            source = hit['_source']
            table_name = source['table_name']
            table_desc = source['table_desc']
            table_descriptions[table_name] = table_desc

        formatted_text = "Table Descriptions:\n"
        for table_name, table_desc in table_descriptions.items():
            formatted_text += f"Table Name: {table_name}\nDescription: {table_desc}\n\n"
        
        today = datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d')
        sys_prompt, usr_prompt = get_prompt_refinement_prompt(table_descriptions, self.language, self.prompt, today)
        response = self.client.converse(
            modelId=self.model,
            messages=usr_prompt,
            system=sys_prompt,
        )
        self.update_tokens(response)
        parsed_json = self.parse_json_format(response['output']['message']['content'][0]['text'])
        self.prompt = parsed_json.get("question")
        return parsed_json

    def query_generation(self, input: str):
        table_names = self.db.get_usable_table_names()
        failure_log = self.failure_log if self.failure_log else 'None'
        failed_query = self.failed_query if self.failed_query else 'None'
        combined_log = f"failure_log: {failure_log}, failed_query: {failed_query}"

        # Get Table Summaries
        if len(table_names) >= 10:
            table_summaries = self.get_table_summaries_by_similarities() # RAG
        else:
            table_summaries = self.get_table_summaries_all()

        # Table Selection
        sys_prompt, usr_prompt = get_table_selection_prompt(table_summaries, self.prompt, self.samples, combined_log)
        response = self.client.converse(
            modelId=self.model,
            messages=usr_prompt,
            system=sys_prompt
        )
        self.update_tokens(response)

        # Loading Table Schemas
        table_names = response['output']['message']['content'][0]['text'].split(',')
        table_schemas = self.get_table_schemas(table_names)
        
        # SQL Query Generation
        sys_prompt, usr_prompt = get_query_generation_prompt(self.dialect, table_schemas, self.language, self.prompt, combined_log)
        response = self.client.converse(
            modelId=self.model,
            messages=usr_prompt,
            system=sys_prompt
        )
        self.update_tokens(response)
        
        parsed_json = self.parse_json_format(response['output']['message']['content'][0]['text'])
        return parsed_json
    
    def validate_and_run_queries(self, generated_query: str):
        explain_query = self.get_explain_query(generated_query)
        try:
            query_plan = self.db.run_no_throw(explain_query)
        except Exception as e:
            return {
                "failure_log": f"An error occurred while generating the EXPLAIN query: {str(e)}",
                "failed_query": generated_query
            }

    
        try:
            sys_prompt, usr_prompt = get_query_validation_prompt(self.dialect, query_plan, generated_query, self.language, self.prompt)
            response = self.client.converse(
                modelId=self.model,
                messages=usr_prompt,
                system=sys_prompt
            )
            self.update_tokens(response)
            
            parsed_json = self.parse_json_format(response['output']['message']['content'][0]['text'])
            query = parsed_json.get("final_query") 
            output_columns = parsed_json.get("output_columns")
        except Exception as e:
            return {
                "failure_log": f"An issue unrelated to the query was encountered: {str(e)} (Model-related problem)",
                "failed_query": query
            }
  
        try:
            result = self.db.run_no_throw(query)
        except Exception as e:
            return {
                "failure_log": f"An error occurred while executing the final query: {str(e)}",
                "failed_query": query
            }
     

        if result is None or (isinstance(result, (list, tuple)) and len(result) == 0):
            return json.dumps({"message": "Query executed successfully, but no matching data found."})
        
        try:
            csv_file, query_file, df = self.save_to_csv(result, output_columns, query)
        except Exception as e:
            return {
                "failure_log": f"An error occurred while saving the results to CSV: {str(e)}",
                "failed_query": query
            }
    
        output = {
            "final_query": query,
            "sql_query_file": query_file,
            "full_result_file": csv_file,
        }
        if len(df) > 20:
            output["partial_result"] = df[:20].to_dict(orient='records')
        else:
            output["full_result"] = df.to_dict(orient='records')

        return output
        
    def tool_router(self, tool, callback):
        with st.spinner(f"Running {tool['name']}..."):
            if tool['name'] == 'prompt_refinement':
                res = self.prompt_refinement(tool['input']['input'])
                tool_result = {"toolUseId": tool['toolUseId'], "content": [{"json": res}]}
            elif tool['name'] == 'query_generation':
                res = self.query_generation(tool['input']['input'])
                tool_result = {"toolUseId": tool['toolUseId'], "content": [{"json": res}]}
            elif tool['name'] == 'validate_and_run_queries':
                self.trial += 1
                res = self.validate_and_run_queries(tool['input']['generated_query'])
                tool_result = {"toolUseId": tool['toolUseId'], "content": [{"json": res}]}
                if 'failure_log' in res:
                    self.failure_log = res['failure_log']
                    self.failed_query = res['failed_query']
                else:
                    self.trial = 0
                    self.failure_log = ""
                    self.failed_query = ""
            else:
                tool_result = {"toolUseId": tool['toolUseId'], "content": [{"text": "Unknown tool name"}]}

        #print("Tool_Result:", res)
        callback.on_llm_new_result(json.dumps({
            "tool_name": tool['name'], 
            "content": tool_result["content"][0]
        }))

        tool_result_message = {"role": "user", "content": [{"toolResult": tool_result}]}

        return tool_result_message

class DB_Tool_Client:
    def __init__(self, model_info, config, language, sql_os_client, schema_os_client, prompt, history):
        self.model = model_info['model_id']
        self.region = model_info['region_name']
        self.dialect = config['dialect']
        self.language = language
        self.top_k = 5
        self.tool_config = self.load_tool_config()
        self.client = self.init_boto3_client(self.region)
        self.tokens = {'total_input_tokens': 0, 'total_output_tokens': 0, 'total_tokens': 0}
        self.prompt = prompt
        self.sys_prompt, self.usr_prompt = get_global_tool_prompt(language, history, prompt)
        self.db_tool = DB_Tools(self.tokens, config['uri'], self.dialect, self.model, self.region, sql_os_client, schema_os_client, language, prompt)

    def init_boto3_client(self, region: str):
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", region_name=region, config=retry_config)

    def load_tool_config(self):
        with open("./db_metadata/tool_config.json", 'r') as file:
            return json.load(file)

    def stream_messages(self, messages, callback):
        response = self.client.converse_stream(
            modelId=self.model,
            messages=messages,
            system=self.sys_prompt,
            toolConfig=self.tool_config
        )

        stop_reason = ""
        message = {"content": []}
        text = ''
        tool_use = {}

        for chunk in response['stream']:
            if 'messageStart' in chunk:
                message['role'] = chunk['messageStart']['role']
            elif 'contentBlockStart' in chunk:
                tool = chunk['contentBlockStart']['start']['toolUse']
                tool_use['toolUseId'] = tool['toolUseId']
                tool_use['name'] = tool['name']
            elif 'contentBlockDelta' in chunk:
                delta = chunk['contentBlockDelta']['delta']
                if 'toolUse' in delta:
                    if 'input' not in tool_use:
                        tool_use['input'] = ''
                    tool_use['input'] += delta['toolUse']['input']
                elif 'text' in delta:
                    text += delta['text']
                    callback.on_llm_new_token(delta['text'])
            elif 'contentBlockStop' in chunk:
                if 'input' in tool_use:
                    tool_use['input'] = json.loads(tool_use['input'])
                    message['content'].append({'toolUse': tool_use})
                    tool_use = {}
                else:
                    message['content'].append({'text': text})
                    text = ''
            elif 'messageStop' in chunk:
                stop_reason = chunk['messageStop']['stopReason']
            elif 'metadata' in chunk:
                self.tokens['total_input_tokens'] += chunk['metadata']['usage']['inputTokens']
                self.tokens['total_output_tokens'] += chunk['metadata']['usage']['outputTokens']
        return stop_reason, message

    def invoke(self, callback):
        messages = self.usr_prompt
        stop_reason, message = self.stream_messages(messages, callback)
        messages.append(message)

        while stop_reason == "tool_use":
            contents = message["content"]
            for c in contents:
                if "toolUse" not in c:
                    continue
                tool_use = c["toolUse"]
                message = self.db_tool.tool_router(tool_use, callback)
                messages.append(message)

            stop_reason, message = self.stream_messages(messages, callback)
            messages.append(message)

        final_response = message['content'][0]['text']
        self.tokens['total_tokens'] = self.tokens['total_input_tokens'] + self.tokens['total_output_tokens']
        return final_response, self.tokens


class InsightTool:
    def __init__(self, filename):
        self.csv_path = filename
        
    def csv_visualizer(self, code: Annotated[str, "The python code to execute to generate your chart."]):
        try:
            result = self.repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
