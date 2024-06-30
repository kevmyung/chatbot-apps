from langchain.prompts.prompt import PromptTemplate

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
- Please generate a valid query without any explanations. 
- For complex questions, do not generate multiple queries, but utilize a compound query as possible.
- Please refer to the samples and schemas to utilize the most relevant table(s).
</instruction>

<response_format>
{{
    "query": "SELECT ... ;",
    "confidence": "Confidence score about the generated SQL query from 0 to 100."
}}
</response_format>
"""

_QUERY_GENERATION_USER_PROMPT = """
<samples>
{samples}
</samples>

<schemas>
{table_schemas}
</schemas>

Previous Known Error: {error_log}
Question: {question}
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

_QUERY_VALIDATION_SYS_PROMPT = """
You are a SQL performance optimization expert of {dialect}. 
Please return the validated query following the instructions.

<instruction>
- Please review the provided query and query plan, and suggest any modifications needed to optimize the query performance.
- Do not introduce new columns or tables. Use only the columns and tables already present in the original query. 
- Add appropriate aliases to tables and columns in the query for improved readability and maintainability.
- Ensure that the final query will conform to the {dialect} syntax.
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

_QUERY_VALIDATION_USER_PROMPT = """
Original Query: {original_query}
Query Plan: {query_plan}
Question: {question}
"""

_FINAL_ANSWER_SYS_PROMPT = """
You are a helpful assistant tasked with efficiently answering user queries in {language}.
Please provide the answer using only the provided context. The response should include the following elements in the specified format:

\n\n--Final Answer--\n
SQL Query: Display the SQL query in a Markdown code block.
Dataframe: Show the resulting dataframe in a table format within a code block. Mention if the result is partial.
Filenames: Include the paths to the result CSV and SQL files in the following format:
  - DataFile: './result_files/query_result_....csv'
  - SQLFile: './result_files/query_....sql'
Answer: Provide a clear and concise answer to the user's question.

If the context does not contain the necessary information to answer the user's query, explain what specific information is missing and refer to the failure log for more details.
Ensure the response is well-organized and easy to follow.
"""

_FINAL_ANSWER_USER_PROMPT = """
<context>
{context}
<context>

Question: {question}
"""

_DB_TOOL_SYS_PROMPT = """
You are a proficient assistant equipped with various tools to handle Text-to-SQL tasks and data acquisition based on user requests. 
Your job is to make progress using the provided tools towards the final results. You will receive the results and messages from each tool. 
Do not actively engage unless the tool returns a failure message and pass it to the next appropriate tools. 
Do not provide the final answer directly.

- Text-to-SQL Task: Convert the user's question into SQL query. Use the 'query_generation' tool for writing the query unless you are over 90% confident based on the sample or context provided in the prompt.
- Query Execution: Validate and execute the generated SQL query using the designated tool. 
- Monitor Progress: Track the progress of each tool's operation and provide the progress at each stage in {language}.
- Schema Exploration: Use it only if the 'validation_and_run_queries' fails due to schema errors such as 'no such table' or 'no such column'.
- Stop Condition: Success message from query execution or failure more than twice across all tools.

Stop all operations as instructed if the Stop Condition is met, even if the task was not successful.
If the task fails, ask more specific table names or column names in the prompt.
"""

_DB_TOOL_USER_PROMPT = """
<Useful_Samples>
{samples}
</Useful_Samples>

Question: {question}
"""

_DATA_FILTERING_SYS_PROMPT = """
You are a skilled assistant specializing in data visualization.
You have been given a large CSV file. Based on the patterns of this data, you need to sample a portion of it to write effective visualization code.
Choose the most appropriate data sampling method based on the given data pattern. The available sampling methods are 'uniform', 'random', 'time_based', and 'sliding_window'.

<response_format>
{{
    "sampling_method": "your choice"
}}
</response_format>
"""

_DATA_FILTERING_USER_PROMPT = """
<data_pattern>
{head}
... ({rows} rows) ...
{tails}
</data_pattern>

User's Request: {question}
"""

_PROMPT_REFINEMENT_SYS_PROMPT = """
You are an expert prompt engineer, and today is {today}.
Your task is to refine the 'original prompt' to achieve a better outcome from the online LLM chatbot.
1. Do your best using the information provided without responding to the user with another request.
2. Do not fabricate any details that are not provided in the context.
3. If the 'original prompt' is contextually linked to the 'conversation history', integrate the context into the 'refined_prompt.'
4. If there are successful SQL queries from a past conversations that are relevant to the current question, include them in the prompt with codeblock.
5. If the requirements are too complex, break it down into smaller, more manageable sub-tasks.
6. Skip the preamble and provide only a well-formed JSON object response.

<response_format>
{{
    "refined_prompt": "Refined prompt"
}}
</response_format>
"""

_PROMPT_REFINEMENT_USER_PROMPT = """
<conversation_history>
{history}
</conversation_history>

original prompt: {question}
"""

_CODE_GENERATION_SYS_PROMPT = """
You are a skilled data visualization engineer specializing in writing plotly python code.
Based on the given CSV data, choose the appropriate chart type for the data and write the visualization code using plotly.
The {datatype}dataframe to be visualized is given within the script. The code should function correctly with the full dataset.

Complete the visualization code for the given Streamlit app. 
Skip any preamble and provide only the code to replace '# Your code here'. Do not include the #--- markers or any other text in your response.
The code should create a Plotly figure named 'fig' using the 'dataframe' that's already loaded. Here's the context:

<visualize.py>
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

dataframe = pd.read_csv(io.StringIO("{dataset}"))
#---
#Your code here
#---
st.plotly_chart(fig)
</visualize.py>

<response_example>
"\n\nfig = px. ...\n"
<response_example>
"""

_CODE_GENERATION_USER_PROMPT = """
<key_columns>
</key_columns>
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

def get_query_generation_prompt(samples, dialect, table_schemas, language, question, error_log):
    return create_prompt(
        _QUERY_GENERATION_SYS_PROMPT,
        _QUERY_GENERATION_USER_PROMPT,
        samples=samples,
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

def get_prompt_refinement_prompt(today, history, question):
    return create_prompt(
        _PROMPT_REFINEMENT_SYS_PROMPT,
        _PROMPT_REFINEMENT_USER_PROMPT,
        history=history,
        question=question,
        today=today
    )

def get_query_validation_prompt(dialect, query_plan, original_query, language, question):
    return create_prompt(
        _QUERY_VALIDATION_SYS_PROMPT,
        _QUERY_VALIDATION_USER_PROMPT,
        dialect=dialect,
        language=language,
        original_query=original_query,
        query_plan=query_plan,
        question=question
    )

def get_global_prompt(language, samples, question):
    return create_prompt(
        _DB_TOOL_SYS_PROMPT,
        _DB_TOOL_USER_PROMPT,
        language=language,
        samples=samples,
        question=question
    )

def get_answer_generation_prompt(language, context, question):
    return create_prompt(
        _FINAL_ANSWER_SYS_PROMPT,
        _FINAL_ANSWER_USER_PROMPT,
        language=language,
        context=context,
        question=question
    )

def get_data_filtering_prompt(question, head, tail, rows):
    return create_prompt(
        _DATA_FILTERING_SYS_PROMPT,
        _DATA_FILTERING_USER_PROMPT,
        question=question,
        head=head,
        tail=tail,
        rows=rows
    )

def get_code_generation_prompt(dataset, datatype):
    return create_prompt(
        _CODE_GENERATION_SYS_PROMPT,
        _CODE_GENERATION_USER_PROMPT,
        dataset=dataset,
        datatype=datatype
    )