# Standard Library Imports
import os
import json
import uuid
import re
import logging
import warnings
import datetime
from typing import List, Dict, Any, Optional, Annotated
from botocore.config import Config
import boto3
import pandas as pd
import ast
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError, SAWarning

# LangChain and Related Imports
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrock
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
from .prompts import get_sql_prompt, get_agent_sys_prompt
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
            sql_os_retriever = OpenSearchHybridRetriever(self.sql_os_client)
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
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3."
            "The output is in JSON format: {{\"{{table_name}}\": {{\"table\": \"{{table_name}}\", \"cols\": {{\"{{col_name}}\": \"{{col_desc}}\",...}}, \"create_table_sql\": \"{{DDL statement for the table}}\", \"sample_rows\": \"{{sample rows for the table}}\"}}}}")
        info_sql_database_tool = CustomInfoSQLDatabaseTool(
            db=self.db, schema_os_client=self.schema_os_client, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a result from the database. "
            "If the query is not correct, an error message will be returned. "
            "If an error is returned, rewrite the query, check the query, and try again. "
            "If you encounter an issue with Unknown column 'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields. "
            "Use this tool as much as possible for generating the final answer. "
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


class ForbiddenQueryError(Exception):
    pass

class DatabaseTool:
    def __init__(self, uri: str, region: str, schema_os_client: OpenSearchClient):
        self.uri = uri
        self.region_name = region
        self.engine = create_engine(uri)
        self.db = SQLDatabase(self.engine)
        self.schema_os_client = schema_os_client
        self.repl = PythonREPLTool()
        if schema_os_client:
            self.table_descriptions = self.load_table_descriptions(schema_os_client)

    def set_database_uri(self, new_uri: str):
        self.uri = new_uri
        self.engine = create_engine(new_uri)
        self.db = SQLDatabase(self.engine)

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
            table_descriptions[table_name] = table_desc
        return table_descriptions

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

    def list_db_tables(self, input: str) -> Dict[str, str]:
        try:
            table_names = self.db.get_usable_table_names()
            if self.schema_os_client:
                tables_dict = {table_name: self.table_descriptions.get(table_name, "No description available") for table_name in table_names}
            else:
                tables_dict = {table_name: "No description" for table_name in table_names}
            
            return tables_dict
        except SQLAlchemyError as e:
            print(f"Error: {e}")
            return {}

    def desc_table_columns(self, table_names: List[str]) -> Dict[str, List[str]]:
        try:
            if any(len(table) == 1 for table in table_names):
                possible_table_name = ''.join(table_names)
                logging.warning(f"Input contains individual characters. Interpreted as '{possible_table_name}'.")
                table_names = [possible_table_name]

            tables = [t.strip() for t in table_names]
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
                if self.schema_os_client:
                    table_desc = self.get_column_description(table, self.schema_os_client)
                else:
                    table_desc = {}
                table_details[table] = {
                    "table": table,
                    "cols": table_desc if table_desc else {},
                    "create_table_sql": sql_statements.get(table, "Not available"),
                    "sample_data": sample_data.get(table, "No sample data available")
                }
                
                if not table_details[table]["cols"]:
                    print(f"No columns found for table {table}")
                
            return table_details
        except OperationalError as oe:
            logging.error(f"OperationalError: {oe}")
            return {"error": "Database operational error occurred. Please check the database connection and try again."}
        except ProgrammingError as pe:
            logging.error(f"ProgrammingError: {pe}")
            return {"error": "Database programming error occurred. Please check the table names and query syntax."}
        except SQLAlchemyError as e:
            logging.error(f"SQLAlchemyError: {e}")
            return {"error": "An unexpected database error occurred. Please try again later."}
        

    def query_checker(self, query: str, dialect: str,  model_id="meta.llama3-70b-instruct-v1:0"):
        chat = ChatBedrock(
            model_id=model_id,
            region_name=self.region_name,
            model_kwargs={"temperature": 0.1},
        )
        message = [
            SystemMessage(
                content=f"""
                Double check the {dialect} query above for common mistakes, including:
                - Using NOT IN with NULL values
                - Using UNION when UNION ALL should have been used
                - Using BETWEEN for exclusive ranges
                - Data type mismatch in predicates
                - Properly quoting identifiers
                - Using the correct number of arguments for functions
                - Casting to the correct data type
                - Using the proper columns for joins

                If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

                Output the final SQL query only.
                """
            ),
            HumanMessage(
                content=query
            )
        ]
        res = chat.invoke(message).content
        return res

    def query_executor(self, query: str, output_columns: List[str]):
        try:
            if re.match(r"^\s*(drop|alter|truncate|delete|insert|update)\s", query, re.I):
                raise ForbiddenQueryError("Sorry, I can't execute queries that can modify the database.")
            
            data = self.db.run_no_throw(query)
            data = ast.literal_eval(data)
            
            if data:
                df = pd.DataFrame(data, columns=output_columns)
                csv_data = df.to_csv(index=False)

                current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                random_id = str(uuid.uuid4())
                folder_path = "./result_files"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                filename = f"{folder_path}/query_result_{current_time}_{random_id}.csv"

                df.to_csv(filename, index=False)
                return {"data": csv_data, "filename": filename}
            else:
                return {"message": "Query executed successfully, but no data was returned."}

        except ForbiddenQueryError as fqe:
            logging.error(f"ForbiddenQueryError: {fqe}")
            return {"error": str(fqe)}
        except (ValueError, SyntaxError) as ve:
            logging.error(f"Data conversion error: {ve}")
            return {"error": "There was an error processing the query results. Please check the query syntax and output columns."}
        except OperationalError as oe:
            logging.error(f"OperationalError: {oe}")
            return {"error": "A database operational error occurred. Please check the database connection and try again."}
        except ProgrammingError as pe:
            logging.error(f"ProgrammingError: {pe}")
            return {"error": "A database programming error occurred. Please check the query syntax and table names."}
        except SQLAlchemyError as e:
            logging.error(f"SQLAlchemyError: {e}")
            return {"error": "An unexpected database error occurred. Please try again later."}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}

    def tool_router(self, tool, callback):
        match tool['name']:
            case 'list_tables':
                res = self.list_db_tables(tool['input']['input'])
                tool_result = {
                    "toolUseId": tool['toolUseId'],
                    "content": [{"json": res}]
                }
            case 'desc_columns':
                res = self.desc_table_columns(tool['input']['tables'])
                tool_result = {
                    "toolUseId": tool['toolUseId'],
                    "content": [{"json": res}]
                }
            case 'query_checker':
                res = self.query_checker(tool['input']['query'], tool['input']['dialect'])
                tool_result = {
                    "toolUseId": tool['toolUseId'],
                    "content": [{"text": res}]
                }
            case 'query_executor':
                res = self.query_executor(tool['input']['query'], tool['input']['output_columns'])
                tool_result = {
                    "toolUseId": tool['toolUseId'],
                    "content": [{"json": res}]
                }
                
        callback.on_llm_new_result(json.dumps({
            "tool_name": tool['name'], 
            "content": tool_result["content"][0]
        }))

        tool_result_message = {"role": "user", "content": [{"toolResult": tool_result}]}

        return tool_result_message

class DatabaseClient_v2:
    def __init__(self, model_info, config, language, sql_os_client, schema_os_client):
        self.model = model_info['model_id']
        self.region = model_info['region_name']
        self.dialect = config['dialect']
        self.language = language
        self.top_k = 5
        self.tool_config = {}
        self.init_client()
        self.sql_os_client = sql_os_client
        self.db_tool = DatabaseTool(config['uri'], self.region, schema_os_client)

    def init_client(self):
        retry_config = Config(
            region_name=self.region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region, config=retry_config)
        with open("./libs/tool_config.json", 'r') as file:
            self.tool_config = json.load(file)

    def get_sample_queries(self, prompt):
        if self.sql_os_client:
            sql_os_retriever = OpenSearchHybridRetriever(self.sql_os_client)
            samples = sql_os_retriever.invoke(prompt, ensemble = [0.51, 0.49])
            return samples
        else:
            return ""

    def stream_messages(self, messages, callback, tokens):
        sys_prompt = get_agent_sys_prompt(self.language)
        response = self.client.converse_stream(
            modelId=self.model,
            messages=messages,
            system=sys_prompt,
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
                tokens['total_input_tokens'] += chunk['metadata']['usage']['inputTokens']
                tokens['total_output_tokens'] += chunk['metadata']['usage']['outputTokens']
        tokens['total_tokens'] = tokens['total_input_tokens'] + tokens['total_output_tokens']
        return stop_reason, message

    
    def invoke(self, prompt, callback):
        tokens = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0
        }

        messages = [{"role": "user", "content": [{"text": prompt}]}]        
        stop_reason, message = self.stream_messages(messages, callback, tokens)
        messages.append(message)

        while stop_reason == "tool_use":
            contents = message["content"]
            for c in contents:
                if "toolUse" not in c:
                    continue
                tool_use = c["toolUse"]
                message = self.db_tool.tool_router(tool_use, callback)
                messages.append(message)

            stop_reason, message = self.stream_messages(messages, callback, tokens)
            messages.append(message)

        final_response = message['content'][0]['text']
        return final_response, tokens
    

class InsightTool:
    def __init__(self, filename):
        self.csv_path = filename
        
    def csv_visualizer(self, code: Annotated[str, "The python code to execute to generate your chart."]):
        try:
            result = self.repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
