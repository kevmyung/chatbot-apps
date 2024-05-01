# from langchain.agents import AgentExecutor, create_xml_agent
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.agents.output_parsers import XMLAgentOutputParser
# from langchain_core.messages import HumanMessage
# from langchain.agents import tool
# import streamlit as st
# from .prompts import get_agent_prompt, get_agent_final_prompt

# class Agent:
#     def __init__(self, llm):
#         self.llm = llm

#         self.message_history = ChatMessageHistory()
#         prompt = get_agent_prompt()
#         tools = self.create_agent_tools(self.llm)
#         self.agent_executor = self.create_agent(prompt, tools)

#     def invoke(self, prompt):

#         response = self.agent_executor.invoke({"input": prompt})
#         # response = self.agent_executor.invoke(
#         #     {"input": prompt}, 
#         #     config={"configurable": {"session_id": st.session_state["widget_key"]}}
#         # )

#         final_prompt = get_agent_final_prompt(response['output'], prompt)
#         return final_prompt

    
#     def convert_intermediate_steps(self, intermediate_steps):
#         log = ""
#         for action, observation in intermediate_steps:
#             log += (
#                 f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
#                 f"</tool_input><observation>{observation}</observation>"
#             )
#         return log
    
#     def convert_tools(self, tools):
#         return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

#     def create_agent(self, prompt, tools):
#         agent = (
#             {
#                 "input": lambda x: x["input"],
#                 "agent_scratchpad": lambda x: self.convert_intermediate_steps(
#                     x["intermediate_steps"]
#                 ),
#             }
#             | prompt.partial(tools=self.convert_tools(tools))
#             | self.llm.bind(stop=["</tool_input>", "</final_answer>"])
#             | XMLAgentOutputParser()
#         )    
#         agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#         return agent_executor
#         # agent_with_chat_history = RunnableWithMessageHistory(
#         #     agent_executor,
#         #     lambda session_id: self.message_history,
#         #     input_messag_key="input",
#         #     history_messages_key="chat_history"
#         # )
#         # return agent_with_chat_history

#     def create_agent_tools(self, llm):
#         @tool
#         def general_answer(prompt: str):
#             "Use this tool when answering questions with general knowledge. It should also be used for responding to simple greetings or basic questions."
#             return "" #answer

#         @tool
#         def answer_with_datbase(prompt: str):
#             "Use this tool for questions that require access to the company database. The database includes tables for artists, albums, media tracks, invoices, and customers."
#             context = st.session_state['db_client'].get_database_context(prompt)
#             return context

#         tools = [general_answer, answer_with_datbase]

#         #tools = [TavilySearchResults(max_results=1)]
#         return tools

