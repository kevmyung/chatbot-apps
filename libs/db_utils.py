from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_xml_agent, tool
from .prompts import get_sql_prompt, get_agent_sys_prompt
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

schema_db = None

def find_sample_queries(os_client, prompt):
    examples = ""
    docs = os_client.vector_store.similarity_search(
        query=prompt, 
        vector_field="input_v", 
        text_field="input",
        score_threshold=0.3
    )
    for doc in docs:
        input_text = doc.metadata.get('input', None)
        query = doc.metadata.get('query', None)
        if input_text and query:
            examples += f"Question: {input_text}\n"
            examples += f"Answer: {query}\n\n"
    return examples


def initialize_schema_db(region_name):
    global schema_db
    schema_db = boto3.resource('dynamodb', region_name=region_name)


def load_table_descriptions(schema_table):
    table = schema_db.Table(schema_table)
    response = table.scan(ProjectionExpression="TableName, Description")
    data = response['Items']
    
    table_descriptions = {}
    for item in data:
        table_name = item['TableName']
        table_desc = item['Description']
        table_descriptions[table_name] = {
            'table_desc': table_desc
        }
    return table_descriptions


def get_table_description(table_name, schema_table):
    table = schema_db.Table(schema_table)
    response = table.get_item(Key={'TableName': table_name})
    if 'Item' in response:
        return response['Item']
    else:
        return None


def initialize_sql_toolkit(db, llm, add_schema_desc, region_name):
    if add_schema_desc:
        initialize_schema_db(region_name)
        return CustomSQLDatabaseToolkit(db=db, llm=llm)
    else:
        return SQLDatabaseToolkit(db=db, llm=llm)


class DatabaseClient:
    def __init__(self, llm, config):
        self.llm = llm
        self.dialect = config['dialect']
        self.add_schema_desc = config['add_schema_desc']
        self.allow_query_exec = config['allow_query_exec']
        self.top_k = 5
        self.region = config['region']
        self.db = SQLDatabase.from_uri(config['uri'])
        self.sql_toolkit = initialize_sql_toolkit(self.db, self.llm, self.add_schema_desc, self.region)
        sql_tools = self.sql_toolkit.get_tools()        
        if self.allow_query_exec == False:
            sql_tools.remove(sql_tools[0])

        prompt = get_sql_prompt()
        agent = create_xml_agent(
            llm=self.llm,
            tools=sql_tools,
            prompt=prompt
        )
        self.sql_executor = AgentExecutor(agent=agent, tools=sql_tools, max_iterations=10) #max_execution_time = 30,


class CustomListSQLDatabaseTool(ListSQLDatabaseTool):
    """Tool for getting tables names."""
    name: str = "custom_sql_db_list_tables"
    description: str = "Input is an empty string, output is a comma-separated list of tables in the database."
    table_descriptions: dict = {}

    def __init__(self, db, schema_table, **kwargs):
        super().__init__(db=db, **kwargs)
        self.table_descriptions = load_table_descriptions(schema_table)

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a dictionary with the structure: 'table_name':'table_description'"""
        table_names = self.db.get_usable_table_names()
        return {table_name: self.table_descriptions.get(table_name, "No description available") for table_name in table_names}


class CustomInfoSQLDatabaseTool(InfoSQLDatabaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "custom_sql_db_schema"
    description: str = "Get the detailed schema and sample rows for the specified SQL tables."
    schema_table: str = "SchemaDescriptions"

    def __init__(self, schema_table, **kwargs):
        super().__init__( **kwargs)
        self.schema_table = schema_table

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
            table_desc = get_table_description(table, self.schema_table)
            if table_desc:
                table_details[table] = {
                    "table": table,
                    "cols": {},
                    "create_table_sql": sql_statements.get(table, "Not available"),
                    "sample_data": sample_data.get(table, "No sample data available")
                }
                for col in table_desc['Columns']:
                    col_name = col['col']
                    col_desc = col['col_desc']
                    table_details[table]['cols'][col_name] = col_desc
            else:
                table_details[table] = {
                    "table": table,
                    "cols": {},
                    "create_table_sql": sql_statements.get(table, "Not available"),
                    "sample_data": sample_data.get(table, "No sample data available")
                }

        return table_details

class CustomSQLDatabaseToolkit(BaseToolkit):
    """
    Toolkit for interacting with SQL databases, now using the custom list tool.
    """
    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    schema_table: str = "SchemaDescriptions"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db, llm, **kwargs):
        super().__init__(db=db, llm=llm, **kwargs)

    def get_tools(self) -> List[BaseTool]:
        list_sql_database_tool = CustomListSQLDatabaseTool(db=self.db, schema_table=self.schema_table)  
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "description about the DB schema and sample rows for those tables. "
            "Use this tool before generating a query. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3."
            "The output is in JSON format: {{\"{{table_name}}\": {{\"table\": \"{{table_name}}\", \"cols\": {{\"{{col_name}}\": \"{{col_desc}}\",...}}, \"create_table_sql\": \"{{DDL statement for the table}}\", \"sample_rows\": \"{{sample rows for the table}}\"}}}}")
        info_sql_database_tool = CustomInfoSQLDatabaseTool(
            db=self.db, schema_table=self.schema_table, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
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
    
