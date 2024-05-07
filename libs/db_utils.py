from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
from langchain.agents import AgentExecutor, create_xml_agent, tool
from .prompts import get_sql_prompt
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
import json
import os

class DatabaseClient:
    def __init__(self, llm, emb, config):
        self.llm = llm
        self.emb = emb
        self.dialect = config['dialect']
        self.top_k = 5
        self.db = SQLDatabase.from_uri(config['uri'])     
        if os.path.exists('libs/schemas.json'):
            sql_toolkit = CustomSQLDatabaseToolkit(db=self.db, llm=self.llm)
        else:
            sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        sql_tools = sql_toolkit.get_tools()
        extra_tools = self.create_agent_tools(sql_tools)

        prompt = get_sql_prompt()
        agent = create_xml_agent(
            llm=self.llm,
            tools=sql_tools+extra_tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(agent=agent, 
                                            tools=sql_tools+extra_tools,
                                            #max_execution_time = 30,
                                            max_iterations=10) 
                                            
        self.sql_chain = create_sql_query_chain(self.llm, self.db)  # not used, but can be used for simple tasks.

    def create_agent_tools(self, input_tools):

        @tool
        def get_today_date(query: str) -> str:
            """
            Use this tool to resolve relative time concepts such as this year, last year, this week, or today by checking today’s date. 
            Input is an empty string, output will be a string in the format %Y-%m-%d.
            """
            import pytz
            today_date_string = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d")
            return today_date_string
        
        @tool
        def parse_dataframe(chat_history: str):
            """
            Useful for extracting and parsing data from chat history.
            """
            # To be implemented

        @tool
        def draw_line_chart(data):
            """
            Use this tool when user asks you to draw line chart. 
            """   
            # To be implemented

        extra_tools = [get_today_date]
        return input_tools + extra_tools


def load_table_descriptions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    table_descriptions = {}
    for item in data:
        for table_name, details in item.items():
            table_desc = details['table_desc'] 
            table_descriptions[table_name] = table_desc
    return table_descriptions


class CustomListSQLDatabaseTool(ListSQLDatabaseTool):
    """Tool for getting tables names."""
    name: str = "custom_sql_db_list_tables"
    description: str = "Input is an empty string, output is a comma-separated list of tables in the database."
    schema_file: str = "./libs/schemas.json"
    table_descriptions: dict = {}

    def __init__(self, db, **kwargs):
        super().__init__(db=db, **kwargs)
        self.table_descriptions = load_table_descriptions(self.schema_file)

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
    schema_file: str = "./libs/schemas.json"
    db_schemas: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open(self.schema_file, 'r') as file:
           data = json.load(file)
           for item in data:
              self.db_schemas.update(item) 

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
            table_details[table] = {
                "table": table,
                "cols": {},
                "create_table_sql": sql_statements.get(table, "Not available"),
                "sample_data": sample_data.get(table, "No sample data available")
            }
            for col in self.db_schemas[table]['cols']:
                col_name = col['col']
                col_desc = col['col_desc']
                table_details[table]['cols'][col_name] = col_desc

        return table_details



class CustomSQLDatabaseToolkit(BaseToolkit):
    """
    Toolkit for interacting with SQL databases, now using the custom list tool.
    """
    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        list_sql_database_tool = CustomListSQLDatabaseTool(db=self.db)  
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "description about the DB schema and sample rows for those tables. "
            "Use this tool before generating a query. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3."
            "The output is in JSON format: {{\"{{table_name}}\": {{\"table\": \"{{table_name}}\", \"cols\": {{\"{{col_name}}\": \"{{col_desc}}\",...}}, \"create_table_sql\": \"{{DDL statement for the table}}\", \"sample_rows\": \"{{sample rows for the table}}\"}}}}")
        info_sql_database_tool = CustomInfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
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