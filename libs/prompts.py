import json
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

_SQL_AGENT_PROMPT = """
You are a helpful assistant tasked with answering user queries efficiently. 
Based on the user's question, compose and validate a {dialect} query, execute the query, and then provide an answer. 
Please provide the final answer including SQL query within a Markdown code block when you are done. 
Final answer should be provided within the tags <final_answer></final_answer>.

You have access to the following tools:
<tools>
{tools}
</tools>
To use a tool, use <tool></tool> and <tool_input></tool_input> tags. 
You will then get back a response in the form <observation></observation>

<example>
If you have a tool called 'sql_db_schema' that loads the table schema, to load the PlayListTrack table schema, respond:

<tool>sql_db_schema</tool><tool_input>PlayListTrack</tool_input>
<observation>
CREATE TABLE "PlaylistTrack" (
    "PlaylistId" INTEGER NOT NULL, 
    "TrackId" INTEGER NOT NULL, 
)
/*
3 rows from PlaylistTrack table:
PlaylistId  TrackId
1   3402
1   3389
1   3390
*/
</observation>
</example>

<chat_history>
{chat_history}
</chat_history>

Refer to the sample queries for query composition if provided.
<samples>
{samples}
</samples>

Begin!

Question: {question}

{agent_scratchpad}
"""

_SQL_AGENT_SYS_PROMPT = """
You are a helpful assistant tasked with efficiently answering user queries.
Utilize the provided tools to progress towards answering the question.
Based on the user's question, compose a SQLite query if necessary, examine the results, and then provide an answer.
If a query fails to execute more than twice, provide a failure message and suggest a retry in your final answer. 
Do not include any hallucinated or fabricated numbers in the results under any circumstances.

Please provide your final answer in {language}.

Make sure to show the below in your final answer:
1. SQL query (Markdown Codeblock)
2. Dataframe (Table in Codeblock)
3. Result Filename (e.g. "\nResult File: './result_files/query_result_....csv'")
4. Answer to the user's question (Text)
"""

_TABLE_SELECTION_SYS_PROMPT = """
You are a data scientist that can help select the most relevant tables for SQL query tasks.
Please select the most relevant table(s) that can be used to generate SQL query for the quetion.

<instruction>
- Skip the preamble and only return the names of most relevant table(s).
- Return at most {top_n} tables.
- Response should be a valid CSV of table names, the format should be "table_name1,table_name2"
</instruction>
"""

_TABLE_SELECTION_USER_PROMPT = """
<samples>
{samples}
</samples>

<table_summaries>
{table_summaries}
</table_summaries>

Previous Known Error - {error_log}
Question: {question}
"""

_QUERY_GENERATION_SYS_PROMPT = """
You are a {dialect} expert.

Please help to generate a {dialect} query to answer the question. 
Your response should ONLY be based on the given context and follow the response guidelines and format instructions.

<instruction>
- If the provided context is sufficient, please generate a valid query without any explanations for the question. 
- For complex questions, do not generate multiple queries, but utilize a compound query as possible.
- Please use the most relevant table(s).
- Ensure the query uses single quotes for string literals and is enclosed in single quotes for the JSON object.
</instruction>

<response_format>
{{
    "query": "SELECT ... ;",
    "confidence": "Confidence score about the generated SQL query from 0 to 100."
}}
</response_format>
"""

_QUERY_GENERATION_USER_PROMPT = """
<schemas>
{table_schemas}
</schemas>

Previous Known Error: {error_log}
Question: {question}
"""

_PROMPT_REFINEMENT_SYS_PROMPT = """
You are a prompt engineering assistant.
Your task is to refine and simplify natural language questions provided by the user for Text-to-SQL tasks. Follow the instructions.

<instruction>
- Please identify the question is ambiguous or too complex.
- If the user's question is too vague or ambiguous, make the question more specific by using the given table descriptions.
- If the user's question is too complex, break it down into smaller, more manageable sub-tasks. Ensure that each sub-task can be easily converted into an SQL query.
- Write a refined user-like qustion in {language}.
- Skip the preamble outside the JSON and provide only a well-formed JSON object response.
</instruciton>

<response_format>
{{
    "question": "A refined question.",
    "complexity": "Set to HIGH if more than three JOIN clauses or nested JOINs are expected to be used; otherwise, set to LOW."
}}
</response_format>

<table_descriptions>
{table_descriptions}
</table_descriptions>
"""

_PROMPT_REFINEMENT_USER_PROMPT = """
Today's date (use only when needed): {today}
User Question: {question}
"""

_SAMPLES_SELECTION_SYS_PROMPT = """
You are a data scientist that can help select the most relevant queries for SQL query tasks.
Please select the most relevant sample question(s) from the provided samples to generate an SQL query for the given question.

<instruction>
- Skip the preamble and only return the indices of the selected samples.
- Select at most {top_n} samples based on similarity to the given question.
- Re-rank the selected sample indices by similarity to the given question.
- Response should be a valid CSV of sample indices (starting from 0), in the format "1,2".
- If no relevant samples are found, return an empty string ("").
</instruction>
"""

_SAMPLES_SELECTION_USER_PROMPT = """
Samples: {samples}
User Question: {question}
"""

_QUERY_VALIDATON_SYS_PROMPT = """
You are a SQL performance optimization expert of {dialect}. 
Please return the validated query following the instructions.

<instruction>
- Please review the provided query and query plan, and suggest any modifications needed to optimize the query performance.
- Do not introduce new columns or tables. Use only the columns and tables already present in the original query. 
- Add appropriate aliases to tables and columns in the query for improved readability and maintainability.
- Provide the modified query along with a brief explanation of the changes made in {language}.
- Skip the preamble outside the JSON and provide only a well-formed JSON object response.
</instruction>

<response_format>
{{
    "final_query": "SELECT ...;",
    "output_columns": ["Column1", "Column2", "Column3"],
    "opinion": "brief explanation for query optimization."
}}
</response_format>
"""

_QUERY_VALIDATON_USER_PROMPT = """
Original Query: {original_query}
Query Plan: {query_plan}
Question: {question}
"""

_GLOBAL_TOOL_SYS_PROMPT = """
You are a helpful assistant tasked with efficiently answering user queries.
Utilize the provided tools to progress towards answering the question.
Based on the user's question, compose a SQL query if necessary, validate and run the results, and then provide an answer.
If a query fails to execute more than twice, provide a failure message and suggest a retry with more specific instruction in your final answer.
Do not include any hallucinated or fabricated numbers in the results under any circumstances.
Please provide your final answer in {language}.

Make sure to include the below in your final answer:
1. SQL query (Markdown Codeblock)
2. Dataframe (Table in Codeblock) : Clarify if it's partial result.
3. Result Data/SQL Filename (e.g. "\n`DataFile: ./result_files/query_result_....csv` \n\n`SQLFile: './result_files/query_....sql`")
4. Answer to the user's question (Text)
"""

_GLOBAL_TOOL_USER_PROMPT = """
<Conversation History>
{history}
</Conversation History>

Question: {question}
"""

def get_sql_prompt():
    return PromptTemplate.from_template(_SQL_AGENT_PROMPT)

def get_agent_sys_prompt(language):
    return [{"text":_SQL_AGENT_SYS_PROMPT.format(language=language)}]

def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

def get_table_selection_prompt(table_summaries, question, samples, error_log):
    return create_prompt(
        _TABLE_SELECTION_SYS_PROMPT,
        _TABLE_SELECTION_USER_PROMPT,
        top_n=10,
        table_summaries=table_summaries,
        question=question,
        samples=samples,
        error_log=error_log
    )

def get_query_generation_prompt(dialect, table_schemas, language, question, error_log):
    return create_prompt(
        _QUERY_GENERATION_SYS_PROMPT,
        _QUERY_GENERATION_USER_PROMPT,
        dialect=dialect,
        language=language,
        table_schemas=table_schemas,
        question=question,
        error_log=error_log
    )

def get_sample_selection_prompt(samples, question):
    return create_prompt(
        _SAMPLES_SELECTION_SYS_PROMPT,
        _SAMPLES_SELECTION_USER_PROMPT,
        top_n=3,
        samples=samples,
        question=question
    )

def get_prompt_refinement_prompt(table_descriptions, language, question, today):
    return create_prompt(
        _PROMPT_REFINEMENT_SYS_PROMPT,
        _PROMPT_REFINEMENT_USER_PROMPT,
        table_descriptions=table_descriptions,
        language=language,
        question=question,
        today=today
    )

def get_query_validation_prompt(dialect, query_plan, original_query, language, question):
    return create_prompt(
        _QUERY_VALIDATON_SYS_PROMPT,
        _QUERY_VALIDATON_USER_PROMPT,
        dialect=dialect,
        language=language,
        original_query=original_query,
        query_plan=query_plan,
        question=question
    )

def get_global_tool_prompt(language, history, question):
    return create_prompt(
        _GLOBAL_TOOL_SYS_PROMPT,
        _GLOBAL_TOOL_USER_PROMPT,
        language=language,
        history=history,
        question=question
    )