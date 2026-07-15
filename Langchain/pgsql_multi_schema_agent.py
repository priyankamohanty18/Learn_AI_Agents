import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor


load_dotenv()
# Create model
openai_key = os.getenv("OPEN_API_KEY")
llm_name = "gpt-3.5-turbo"
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

postgres_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database_name}?options=-csearch_path=raw_schema,prod_schema"
engine = sqlalchemy.create_engine(postgres_uri)

with engine.connect() as conn:
    result = conn.execute(
        text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema IN ('raw_schema', 'prod_schema') AND table_type='BASE TABLE';
        """
        )
        
    )
    tables = [row[0] for row in result.fetchall()]

print(tables)

db = SQLDatabase.from_uri(
    postgres_uri,
    include_tables=tables,
    sample_rows_in_table_info=10
)


#db = SQLDatabase.from_uri(postgres_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=model)

QUESTION = """what is the address of Joe Bloggs?"""

#QUESTION = """who are active customers?"""
#QUESTION = """what is the max salary of employees?"""
#QUESTION = """who is the software engineer?"""


#QUESTION = """how many records are present in products table?"""

sql_agent = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
)

res = sql_agent.invoke(QUESTION,handle_parsing_errors=True)
# res = sql_agent.invoke(QUESTION)

print(res)
