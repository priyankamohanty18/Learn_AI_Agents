import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor,initialize_agent, Tool



load_dotenv()
# Create model
openai_key = os.getenv("OPEN_API_KEY")
llm_name = "gpt-3.5-turbo"
#llm_name = "gpt-4-turbo"
model = ChatOpenAI(api_key= openai_key, model= llm_name)


# Prepare the sql pre and post prompt
MSSQL_AGENT_PREFIX = """
You are an agent designed to interact with a SQL database.

## Instructions:
- Given an input question, create a syntactically correct {dialect} query to run.
- Always limit your query to at most {top_k} results unless specified.
- Never query all columns (*), always specify the required columns.
- You MUST execute the query first before giving a final answer.
- Your response should be in Markdown. (No backticks inside SQL inputs)
- Only use tables you find in the database schema.
- Before providing any response, scan through the whole table to check if the data records are present.
- If unsure, say "I don't know" instead of guessing.
"""


MSSQL_AGENT_FORMAT_INSTRUCTIONS = """
## Use this format:

Question: the input question you must answer.
Thought: reason about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the SQL query to run (no backticks).
Observation: result of running the query.
... (loop Thought → Action → Observation if needed)
Thought: I now know the final answer.
Final Answer: the final answer to the question.

At the end, explain how you got the answer in a short paragraph.
"""

# Path to your postgresql database
host = "localhost"
port = 5432
user = "agent_user"
password = "agent_pwd"
database_name = "mydb"

postgres_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database_name}"

db_raw = SQLDatabase.from_uri(
    postgres_uri,
    include_tables=[ "products","customers","employees","cust_detail","inventory_movements","order_items"],
    sample_rows_in_table_info=3,           
    schema="raw_schema"                    
)

db_prod = SQLDatabase.from_uri(
    postgres_uri,
    include_tables=["customers"],
    sample_rows_in_table_info=3,           
    schema="prod_schema"                    
)

#db = SQLDatabase.from_uri(postgres_uri)
raw_toolkit = SQLDatabaseToolkit(db=db_raw, llm=model)
prod_toolkit = SQLDatabaseToolkit(db=db_prod, llm=model)

sql_raw_agent = create_sql_agent(
    llm=model,
    toolkit=raw_toolkit,
    verbose=True,
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
)

sql_prod_agent = create_sql_agent(
    llm=model,
    toolkit=prod_toolkit,
    verbose=True,
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
)

# Define Tools
tools = [
    Tool(
        name="SQL_DB_raw_schema",
        func=sql_raw_agent.run,
        description= """Use this tool to query structured data from the raw_schema in postgresql database. 
          If some columns are not available in the raw_schema in SQL database, ensure that the 
    missing data is queried from the other tools to generate a complete response."""
    ),
    Tool(
        name="SQL_DB_prod_schema",
        func=sql_prod_agent.run,
        description= """ Use this tool to query structured data from the prod_schema in postgresql database. 
          If some columns are not available in the prod_schema in SQL database, ensure that the 
    missing data is queried from the other tools to generate a complete response."""
    )
]

# Define system prompt for the agent
system_prompt = """
You are an intelligent assistant capable of querying multiple schemas in the postgresql database to generate insights.
- For each query, first identify all the required columns.
- Carefully check the query and determine the table and the schema in the postgresql database,which can provide the required information 
and use the relevant tool.
- Carefully identify the table which has the required data records and columns in each schema in postgresql database.
- Scan through the whole table to check if the data records are present.Do not give any results without scanning the whole table.
- If you do not find the data records in the 'SQL_DB_raw_schema' tool, then check the 'SQL_DB_prod_schema' tool.
- Treat natural language terms as synonyms of actual column names.For eg, "phone number" and "contact number" refer to a column named "phone". 
- When the user queries about a product name or an employee or a customer or any Proper noun, convert the product name, employee name, 
  customer name or any Proper noun into lower case and apply the lower case on the value in the corresponding columns in the database tables, 
  before comparing to ensure comparison is case-insensitive. 
  For example: 
  User query : what is the phone number of Diana rose?
  SQL query to be formed : SELECT phone FROM customers WHERE lower(name) = lower('Diana Rose');
  Reponse: {'input': 'what is the phone number of Diana Rose?', 'output': 'The phone number of Diana Rose is 555-4321.'}
- A person could be either an employee or a customer and there could be different people with the same name, existing in separate tables.
  Whenever a user queries about the details of a person, check it in all the tables in raw_schema and prod_schema and return
  all the possible values for that person. For eg; If there is a person with name, 'Diana Rose' in the customers table and there is a different
  person with the same name,'Diana Rose' in the employees table.Please return the details about Diana Rose from both the tables. 
- Follow the next set of instructions carefully .
- Use the 'SQL_DB_raw_schema' tool to query structured data stored in the raw_schema schema in postgresql database. 
- Use the 'SQL_DB_prod_schema' tool to query structured data stored in the prod_schema schema in postgresql database. 
- If any of the records or columns or tables are not present in the 'SQL_DB_raw_schema' tool, check the 'SQL_DB_prod_schema' tool for the columns and tables.
- Given an input question, create a syntactically correct query to run based on the data records and columns present in the 'SQL_DB_raw_schema' tool and
and do not add any column in the SQL query which you do not find in tables in the 'SQL_DB_raw_schema' tool.
- If you do not find the data records or columns or tables in 'SQL_DB_raw_schema' tool,then check for the data records, columns and tables in the 'SQL_DB_prod_schema' tool and generate the SQL query for the data records,columns and tables which are present in the 'SQL_DB_prod_schema' tool.
- Provide a well-formatted and complete response that integrates the data.
"""

# Initialize multi-tool agent
agent_executor = initialize_agent(
    tools, llm = model, verbose=True,agent_kwargs={"system_message": system_prompt},handle_parsing_errors=True
)

#res = agent_executor.invoke("what is the address of Joe Bloggs?")
#res = agent_executor.invoke("what is the phone number of Bob Smith?")
res = agent_executor.invoke("what is the phone number of Diana Rose?")
#res = agent_executor.invoke("what is the phone number of employee,Diana Rose?")
#res = agent_executor.invoke("what is the phone number of customer,Diana Rose?")
#res = agent_executor.invoke("what is the phone number of customer,Diana Rose and employee,Diana Rose?")
#res = agent_executor.invoke("what is the phone number of Diana rose?")
#res = agent_executor.invoke("what is the phone number of John Davies?")
#res = agent_executor.invoke("what is the contact number of Bob Smith?")
#res = agent_executor.invoke("How do I contact Eve Adams?")

#res = agent_executor.invoke("what is the price of item - chair?")
#res = agent_executor.invoke("What is the price of item,desk chair?")
#res = agent_executor.invoke("What is the price of item,Desk Chair?")
#res = agent_executor.invoke("How much do Headphones cost?")
#res = agent_executor.invoke("What is the price of headphone?")
#res = agent_executor.invoke("What is the price of desk chair?")
#res = agent_executor.invoke("What is the cost of a Laptop?")



print(res)

#QUESTION = """what is the address of Joe Bloggs?"""
#QUESTION = """what is the phone number of John Davies?"""


#QUESTION = """what is the price of desk chair?"""
#QUESTION = """who is the Software Engineer?"""
#QUESTION = """who are active customers?"""
#QUESTION = """what is the max salary of employees?"""



#QUESTION = """how many records are present in products table?"""
