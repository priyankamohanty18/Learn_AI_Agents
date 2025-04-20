
import pandas as pd
import sqlite3
from autogen import AssistantAgent, UserProxyAgent, GroupChat,GroupChatManager,register_function
from dotenv import load_dotenv
import os

load_dotenv()

model = "gpt-3.5-turbo"
llm_config ={
    "model" : model,
    "api_key" : os.getenv("OPENAI_API_KEY")
}

file_path = './prod.csv'
db_path = './db_autogen.db'
table_name = 'product'


def read_and_insert_csv(file_path: str, db_path: str, table_name: str):
    print("read_and_insert_csv called")
    try:
        df = pd.read_csv(file_path)
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        return f"Inserted {len(df)} records from '{file_path}' into '{table_name}'"
    except Exception as e:
        return f"Failed to load CSV into DB: {e}"
    

loader = AssistantAgent(
    name="CSV_Loader_Agent",
    description="reads and loads data from CSV file into a SQLite database",
    llm_config=llm_config,
    #system_message= f"You are a database loader who loads data into SQLite.You must always call the function {read_and_insert_csv(file_path: str, db_path: str, table_name: str)}",
    system_message="You are a database loader. Use the registered function to read and load data into SQLite.You must always call the function `read_and_insert_csv(file_path: str, db_path: str, table_name: str)` using the registered function interface.",
    code_execution_config = {
    "use_docker": False
    }
)


loader.register_function({"read_and_insert_csv": read_and_insert_csv})

# Send user message that matches what the function expects
#response = loader.generate_reply(
#    messages = [
#        {
#            "role": "user",
#           # "content": f"Please load data using read_and_insert_csv('{file_path}', '{db_path}', '{table_name}')"
#            "content": f"Call `read_and_insert_csv('{file_path}', '{db_path}', '{table_name}')`"
#        }
#    ]
#)

response = loader.generate_reply(
    messages=[
        {
            "role": "user",
            "content": None,
            "function_call": {
                "name": "read_and_insert_csv",
                "arguments": f'''{{
                    "file_path": "{file_path}",
                    "db_path": "{db_path}",
                    "table_name": "{table_name}"
                }}'''
            }
        }
    ]
)
print(response)

