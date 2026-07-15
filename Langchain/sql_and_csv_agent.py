import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


load_dotenv()

openai_key = os.getenv("OPEN_API_KEY")
#llm_name = "gpt-3.5-turbo"
llm_name = "gpt-4o"
model = ChatOpenAI(api_key= openai_key, model= llm_name)


# Create an engine to connect to the SQLite database
# SQLite only requires the path to the database file
#engine = create_engine(f"sqlite:///{database_file_path}")

# Part 2: Prepare the sql pre and post prompt
MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct query
to run, then look at the results of the query and return the answer.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.
- as part of your final answer, please include the SQL query you used in json format or code format

## Tools:

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT max ([amount])
FROM Payments

WHERE payment_date > '2024-03-01 10:05:00'

Observation:
[(109.98)]
Thought:I now know the final answer
Final Answer: The highest payment amount from Payments table where payment_date > '2024-03-01 10:05:00' is 109.98.

Explanation:
I queried the `Payments` table for the `amount` column where the payment_date
is greater than '2024-03-01 10:05:00'. The query returned a list of tuples
with the amount for each day. To answer the question,
I took the highest of all the amounts in the list, which is 109.98.

===> End of Example

"""
# Path to your SQLite database file
database_file_path = "./new_db.db"
db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

QUESTION = """which is the order with the highest number of items and what is the total quantity of items?"
"""
sql_agent = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    verbose=True,
)

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, Tool

#read csv file
df = pd.read_csv('./data/user.csv').fillna(value=0)
#create csv agent
csv_agent = create_pandas_dataframe_agent(llm = model, df = df, verbose = True, allow_dangerous_code= True)

# Define Tools
tools = [
    Tool(
        name="SQL Database",
        func=sql_agent.run,
        description= """Use this tool to query structured data from the SQL database. 
          If some columns are not available in the SQL database, ensure that the 
    missing data is queried from the CSV file to generate a complete response."""
    ),
    Tool(
        name="CSV Data",
        func=csv_agent.run,
        description= """ Use this tool to analyze and extract data from the CSV file. 
    This tool is ideal when information is not found in the SQL Database, 
    or when data needs to be aggregated or analyzed from tabular data. 
    If the query mentions any fields that do not exist in the SQL database, 
    use this tool to look them up and merge the results with the database query"""
    )
]

# Define system prompt for the agent
system_prompt = """
You are an intelligent assistant capable of querying both a SQL database and a CSV file to generate insights.
- For each query, first identify all the required columns.
- First check if a query requires information from both sources,i.e, 'SQL Database' tool and 'CSV Data' tool.Follow the next set of instructions carefully .
- Use the 'SQL Database' tool to query structured data stored in the SQL database. 
-Given an input question, create a syntactically correct query to run based on the columns present in the 'SQL Database' tool. 
and do not add any column in the SQL query which you do not find in tables in the 'SQL Database' tool.Do note that some of the 
columns may be present in the 'CSV Data' tool.Generate the SQL query for the columns which are present in the 'SQL Database' tool.
- If any of the columns are not present in the 'SQL database' tool, check the 'CSV Data' tool for the columns which are missing in 'SQL database' tool.
- Once you find the required columns in the 'CSV Data' tool, then think carefully and read the input query again and intelligently combine the output from the 'SQL database' tool with the output from the 'CSV Data' tool. 
- Provide a well-formatted and complete response that integrates the data.
"""


# Initialize multi-tool agent
agent_executor = initialize_agent(
    tools, llm = model, verbose=True,agent_kwargs={"system_message": system_prompt},handle_parsing_errors=True
)

res = agent_executor.invoke("find the user who made the payment with the highest amount for an order and provide the phone number of the user?")
# res = sql_agent.invoke(QUESTION)
# res = sql_agent.invoke(QUESTION)

print(res)
