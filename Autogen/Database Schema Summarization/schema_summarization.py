
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

db_path = "./db_autogen.db"
db = "db_autogen.db"

connection_string = sqlite3.connect(db_path)

def summarize_database():
    try:  
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        tables = cursor.fetchall()
        
        if not tables:
            return "No tables found in the database."

        table_summary = []
        table_relationships = []

        for table in tables:
            table_name = table[0]

            # Get columns for the table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            table_summary.append(f"\n Table: {table_name}")
            for column in columns:
                table_summary.append(f"  - {column[1]} ({column[2]})")

            # Get foreign key relationships for the table
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            for fk in foreign_keys:
                table_relationships.append(
                    f"{table_name}.{fk[3]} → {fk[2]}.{fk[4]}"
                )

        conn.close()

        result = "\n".join(table_summary)
        if table_relationships:
            result += "\n Foreign Key Relationships:\n" + "\n".join(table_relationships)
        else:
            result += "\n(No foreign key relationships found.)"

        return result

    except Exception as e:
        return f"Error: {e}"
     
        
#assistant = AssistantAgent( 
#               name= "DB_Summarizer_Agent",
#               description="Reads the sqlite database and summarizes the details about the tables and relationships between tables",
#               system_message= f"You are an assistant who summarizes the sqlite database.Read the database, {db_path} at the {connection_string} and understand the database schema, tables , relationship between tables and summarize the details",
#               llm_config=llm_config
#    )


assistant = AssistantAgent( 
    name="DB_Summarizer_Agent",
    description="Summarizes the structure of the sqlite database using a registered function.",
    system_message="""
You are an assistant that summarizes the SQLite database structure using a Python function named `summarize_database()`.

DO NOT make up schema or table names. DO NOT write code blocks.
When the user asks to summarize the database, you must respond by calling the function like this:

summarize_database()

Do not explain or modify the function. Just invoke it.
""",
    llm_config=llm_config
)


user_proxy = UserProxyAgent(
    name="User",
    llm_config=llm_config,
    code_execution_config= {"work_dir": "my_code_dir",
                            "use_docker": False},
    human_input_mode= "NEVER"
)

user_proxy.register_function({"summarize_database": summarize_database})

#user_proxy.initiate_chat(
#    recipient=assistant,
#    message="Summarize the structure of sqlite database",
#    max_turns=3
#)

response = user_proxy.generate_reply(
    messages=[
        {
            "role": "user",
            "content": None,
            "function_call": { "name": "summarize_database"}
        }  
             ]                                 
                                    )
print(response)