from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_xml_agent, tool
from .prompts import get_sql_prompt
from .opensearch import OpenSearchHybridRetriever, OpenSearchClient
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.tools import BaseToolkit
from typing import Any, List, Optional
import boto3
import json
import warnings
from sqlalchemy.exc import SAWarning

warnings.filterwarnings("ignore", r".*support Decimal objects natively, and SQLAlchemy", SAWarning)

class DatabaseClient:
    def __init__(self, llm, config, sql_os_client, schema_os_client):
        self.llm = llm
        self.dialect = config['dialect']
        self.top_k = 5
        self.sql_os_client = sql_os_client
        self.schema_os_client = schema_os_client
        self.allow_query_exec = config['allow_query_exec']
        self.db = SQLDatabase.from_uri(config['uri'])
        self.sql_toolkit = self.initialize_sql_toolkit()
        sql_tools = self.sql_toolkit.get_tools()        
        if self.allow_query_exec == False:
            sql_tools.remove(sql_tools[0])

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
    
