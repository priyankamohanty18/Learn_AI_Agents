
import pandas as pd
import sqlite3
from autogen import AssistantAgent, UserProxyAgent,register_function
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
   
def delete_product(db_path: str, table_name: str, product_id: int):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name} WHERE Product_ID = ?", (product_id,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return f"Deleted {rows_affected} records from the '{table_name}' table"
    except Exception as e:
        return f"Failed to delete product: {e}"    


remover = AssistantAgent(
    name="CSV_Delete_Agent",
    description="deletes data from table in SQLite database",
    llm_config=llm_config,
    #system_message= f"You are a database loader who loads data into SQLite.You must always call the function {read_and_insert_csv(file_path: str, db_path: str, table_name: str)}",
    system_message="You are an assistant who deletes data from SQLite. Use the registered function to delete data in SQLite.You must always call the function `delete_product(db_path: str, table_name: str, product_id: int)` using the registered function interface.",
    code_execution_config = {"use_docker": False}
)

remover.register_function({"delete_product": delete_product})

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


# Example: Delete product with Product_ID = 102
response_delete = remover.generate_reply(
    messages=[
        {
            "role": "user",
            "content": None,
            "function_call": {
                "name": "delete_product",
                "arguments": f'''{{
                    "db_path": "{db_path}",
                    "table_name": "{table_name}",
                    "product_id": 102
                }}'''
            }
        }
    ]
)

print(response_delete)

