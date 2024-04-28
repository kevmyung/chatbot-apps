from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.agents import AgentExecutor, create_xml_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from .prompts import get_sql_prompt, get_sql_final_prompt
from langchain.agents import tool
from langchain_community.tools.tavily_search import TavilySearchResults

class DatabaseClient:
    def __init__(self, llm, emb, config):
        self.llm = llm
        self.emb = emb
        self.dialect = config['dialect']
        self.top_k = 5
        self.db = SQLDatabase.from_uri(config['uri'])
        self.handler = config['handler']
        self.table_schema = self.db.get_context()['table_info']
        self.table_name = self.db.get_context()['table_names']

        if self.handler == 'SQL Chain':
            self.sql_chain = create_sql_query_chain(self.llm, self.db)
            sql_chain_prompt = get_sql_prompt(prompt_type='sql_chain', db_client=self)
            self.sql_chain.get_prompts()[0].template = sql_chain_prompt
        elif self.handler == 'Simple SQL Chain' or self.handler == 'Agent':
            sql_chain_prompt = get_sql_prompt(prompt_type='simple_sql_chain', db_client=self)
            self.sql_chain = self.create_simple_db_chain(sql_chain_prompt)
        elif self.handler == 'SQL Agent':
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            tools = toolkit.get_tools()
#            db_tools = toolkit.get_tools()
#            search_tools = [TavilySearchResults(max_results=1)]
#            tools = db_tools + search_tools

            sql_agent_prompt = get_sql_prompt(prompt_type='sql_agent', db_client=self) 
            self.agent = create_xml_agent(llm=self.llm, tools=tools, prompt=sql_agent_prompt)
            self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
        else:
            raise ValueError("Unsupported SQL handler type specified in configuration") 
    
    def create_simple_db_chain(self, prompt):
        db_chain = SQLDatabaseChain.from_llm( 
            llm=self.llm,
            db=self.db,
            prompt=prompt,
            verbose=True,
            use_query_checker=True
        )
        return db_chain

    def run_queries_with_chain(self, prompt):
        sql_response = self.sql_chain.invoke({"question": prompt})
        if sql_response == "InvalidRequest":
            sql_result = "No information available for the provided database."
        else: 
            sql_result = self.db.run(sql_response)
        
        context_template = """
            <question>
            {prompt}
            </question>

            <sql_query>
            {sql_response}
            </sql_query>

            <sql_result>
            {sql_result}
            </sql_result>
        """
        
        return context_template.format(prompt=prompt, sql_response=sql_response, sql_result=sql_result)
    
    def run_queries_with_agent(self, prompt):
        response = self.agent_executor.invoke({"input": prompt})
        
        context_template = """
            <final_answer>
            {final_answer}
            </final_answer>
        """        

        return context_template.format(final_answer=response['output'])

    def get_context_from_chain(self, prompt):
        context_text = self.run_queries_with_chain(prompt)
        final_prompt = get_sql_final_prompt(context_text, prompt)
        return final_prompt

    def get_context_from_simple_chain(self,prompt):
        context_text = self.sql_chain.invoke(prompt)
        final_prompt = get_sql_final_prompt(context_text, prompt)
        return final_prompt

    def get_context_from_agent(self, prompt):
        context_text = self.run_queries_with_agent(prompt)
        final_prompt = get_sql_final_prompt(context_text, prompt)
        return final_prompt
    
    def get_database_context(self, prompt):
        if self.handler == 'SQL Chain':
            return self.get_context_from_chain(prompt)
        if self.handler == 'Simple SQL Chain' or self.handler == 'Agent':
            return self.get_context_from_simple_chain(prompt)
        elif self.handler == 'SQL Agent':
            return self.get_context_from_agent(prompt)
        else:
            raise ValueError("Unsupported SQL handler type specified in configuration")             
