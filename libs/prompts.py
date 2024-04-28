import json
from langchain import hub
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS



_FINAL_PROMPT = """
Note: If the 'context' provides answers based on general knowledge, respond kindly to the user's questions. Don't include XL tags. Don't mention 'context'
Note: If the 'context' provides answers related to the database, respond according to the instructions between <sql_instruction></sql_instruction>'.
"""

_SQL_FINAL_PROMPT = """
1. 답변 
   - Directly provide the answer to the user's question using a dataframe with detailed numerical data. Place this detailed answer at the beginning of the response.
2. 부가 설명 
   - In a separate section, explain in Korean which database tables were utilized and the reasons behind joining or filtering certain tables. Provide clear explanations to help users understand the data processing logic and decisions.
3. 사용된 SQL 쿼리 
   - Lastly, include the actual SQL query that was used, formatted neatly within a code block. Place this query at the end of the Data Processing Explanation Section. Provide a brief description in Korean above the query to clarify what the query accomplishes and include any necessary comments within the code to assist with understanding.

Note: Ensure each section is distinctly separated and clearly labeled to facilitate an easy understanding of the response. 
Note: Ensure no XML tags are included in the answer.
Note: Please write all explanations in Korean.
"""

_SIMPLE_SQL_CHAIN_PROMPT="""
Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

<Question>"Question here"</Question>
<SQLQuery>"SQL Query to run"</SQLQuery>
<SQLResult>"Result of the SQLQuery"</SQLResult>
<Answer>"Final answer here"</Answer>

Only use the tables listed below.

{table_info}

Question: {input}`

Provide Question, SQLQuery, SQLResult, Answer with XML tags.
"""

_SQL_CHAIN_PROMPT="""
You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today". 

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
To do so, first explain which tables you need to access. Then determine what you need to join (and why), and what you need to filter on (and why). Only after that construct the query.
If the question does not seem related to the database, just return "InvalidRequest" as the answer.

Only use the following tables:
<table_name>
{table_name}
</table_name>

<table_info>
{table_schema}
</table_info>

Question: {input}

Skip the preamble and provide only the SQL.
"""

_HYBRID_AGENT_PROMPT="""
You are a helpful assistant. Help the user answer any questions.
You can generate a syntatically correct {dialect} query and execute it for answers, or resort to information obtained via web searches.
You can provide ageneral response without resorting to querying information, even without using tools.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

if you have a tool called 'sql_db_list_tables' that could retrieve the list of Database tables, in order to get the list you would respond:
<tool>sql_db_list_tables</tool><tool_input></tool_input>
<observation>Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:
<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

Previous Conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""


_SQL_AGENT_PROMPT_PREFIX="""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:
"""

_SQL_AGENT_PROMPT_SUFFIX="""
In this environment you have access to a set of tools you can use to answer the user's question.
You may call them like this:
<tools>
{tools}
</tools>

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. 
You will then get back a response in the form <observation></observation>.
For example, if you have a tool called 'sql_db_schema' that could retrieve Database schema, in order to describe the playlisttrack table you would respond:

<tool>sql_db_list_tables</tool><tool_input></tool_input>

<observation>Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track</observation>

<tool>sql_db_schema</tool><tool_input>PlatListTrack</tool_input>

<observation>
CREATE TABLE "PlaylistTrack" (
    "PlaylistId" INTEGER NOT NULL, 
    "TrackId" INTEGER NOT NULL, 
    PRIMARY KEY ("PlaylistId", "TrackId"), 
    FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
    FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)
/*
3 rows from PlaylistTrack table:
PlaylistId  TrackId
1   3402
1   3389
1   3390
*/
</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>
Here is the schema of the `PlaylistTrack` table:
```
CREATE TABLE "PlaylistTrack" (
    "PlaylistId" INTEGER NOT NULL, 
    "TrackId" INTEGER NOT NULL, 
    PRIMARY KEY ("PlaylistId", "TrackId"), 
    FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
    FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)
```
The `PlaylistId` column is a foreign key referencing the `PlaylistId` column in the `Playlist` table. 
The `TrackId` column is a foreign key referencing the `TrackId` column in the `Track` table.
Here are three sample rows from the `PlaylistTrack` table:
```
PlaylistId   TrackId
1            3402
1            3389
1            3390
```
</final_answer>

Begin!

Question: {input}

Let's think step by step. {agent_scratchpad}
"""


_AGENT_PROMPT = """
You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

Previous Conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""


def nonfailing_format(template, **kwargs):
    class Default(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(Default(kwargs))

def get_sql_prompt(prompt_type, db_client):
    if prompt_type == 'sql_chain':
        data = {"dialect": db_client.dialect, "top_k": db_client.top_k, "table_schema": db_client.table_schema, "table_name": db_client.table_name}
        return nonfailing_format(_SQL_CHAIN_PROMPT, **data)  
    elif prompt_type == 'simple_sql_chain':
        data = {"dialect": db_client.dialect, "top_k": db_client.top_k, "table_info": db_client.table_schema}
        prompt = nonfailing_format(_SIMPLE_SQL_CHAIN_PROMPT, **data)  
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template = prompt
        )
        return prompt_template
    elif prompt_type =='sql_agent':
        data = {"dialect": db_client.dialect, "top_k": db_client.top_k}
        AGENT_PROMPT_PREFIX = nonfailing_format(_SQL_AGENT_PROMPT_PREFIX, **data)
        agent_prompt = create_few_shot_prompt(db_client.emb, AGENT_PROMPT_PREFIX, _SQL_AGENT_PROMPT_SUFFIX)
        #agent_prompt = nonfailing_format(_HYBRID_AGENT_PROMPT, **data)
        return agent_prompt

def create_few_shot_prompt(emb, _PREFIX, _SUFFIX):
    example_prompt = PromptTemplate(
        input_variables=["input", "query"],
        template="User input: {input}\nSQL query: {query}"
    )

    with open('./libs/examples.json', 'r') as file:
        examples = json.load(file)

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        emb,
        FAISS,
        k=5,
        input_keys=["input"],
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_PREFIX,
        suffix=_SUFFIX,
        input_variables=["input", "query"]
    )
    return few_shot_prompt

def get_sql_final_prompt(context_text, prompt):
    return f"{_SQL_FINAL_PROMPT}\n\nHere is some context for you: \n<context>\n{context_text}</context>\n\n{prompt}"

def get_agent_final_prompt(context_text, prompt):
    return f"{_FINAL_PROMPT}\n<sql_instruction>{_SQL_FINAL_PROMPT}</sql_instruction>\n\nHere is some context for you: \n<context>\n{context_text}</context>\n\n{prompt}"

def get_agent_prompt():
    prompt_template = PromptTemplate(
        input_variables=["agent_scratchpad", "chat_history", "input", "tools"],
        template=_AGENT_PROMPT
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=prompt_template
    )
    prompt = ChatPromptTemplate(
        input_variables=["agent_scratchpad", "input", "tools"],
        partial_variables={"chat_history": ""},
        messages=[human_message_prompt]
    )
    return prompt