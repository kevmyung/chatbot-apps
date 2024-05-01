from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
from langchain.agents import AgentExecutor, create_xml_agent, tool
from langchain.memory import ChatMessageHistory
from .prompts import get_sql_prompt
from langchain_core.tools import ToolException
import pandas as pd
import numpy as np
import streamlit as st


class DatabaseClient:
    def __init__(self, llm, emb, config):
        self.llm = llm
        self.emb = emb
        self.dialect = config['dialect']
        self.top_k = 5
        self.db = SQLDatabase.from_uri(config['uri'])     
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        sql_tools = sql_toolkit.get_tools()
        #extra_tools = self.create_agent_tools(sql_tools)

        prompt = get_sql_prompt()
        agent = create_xml_agent(
            llm=self.llm,
            tools=sql_tools,
            #tools=sql_tools+extra_tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(agent=agent, 
                                            tools=sql_tools,
                                            #tools=sql_tools+extra_tools,
                                            #max_execution_time = 30,
                                            max_iterations=10) 
                                            
        self.sql_chain = create_sql_query_chain(self.llm, self.db)  # not used, but can be used for simple tasks.

    def create_agent_tools(self, input_tools):

        @tool
        def get_today_date(query: str) -> str:
            """
            Use this tool to get the date of today.
            """
            today_date_string = datetime.now().strftime("%Y-%m-%d")
            return today_date_string 
        
        @tool
        def parse_dataframe(chat_history: str):
            """
            Useful for extracting and parsing data from chat history.
            """
            # To be implemented

        @tool(return_direct=True)
        def draw_line_chart(data: List):
            """
            Use this tool when user asks you to draw line chart. 
            """   
            # To be implemented

        extra_tools = [get_today_date, draw_line_chart]
        return input_tools + extra_tools
            