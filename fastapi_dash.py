#Importing the required Libraries
import json
import pandas as pd
#from test_chart import app as dash_app
import requests
from fastapi.middleware.wsgi import WSGIMiddleware
#import plotly.express as px
import _jsonnet
import psycopg2
import re
import torch
from seq2struct.commands.infer import Inferer
from seq2struct.datasets.spider import SpiderItem
from seq2struct.utils import registry
from fastapi import Request
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from test_chart import app_layout

#Configuration Loading
exp_config = json.loads(
    _jsonnet.evaluate_file(
    "experiments/spider-configs/gap-run.jsonnet"))
model_config_path = exp_config["model_config"]
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(
    _jsonnet.evaluate_file(
        model_config_path,
        tla_codes={'args': json.dumps(model_config_args)}))
infer_config["model"]["encoder_preproc"]["db_path"] = "data/sqlite_files/"

#Inference on the input
inferer = Inferer(infer_config)

#Pre-Trained Model checkpoints Loading
model_dir = exp_config["logdir"] + "/bs=12,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1"
checkpoint_step = exp_config["eval_steps"][0]

model = inferer.load_model(model_dir, checkpoint_step)

from seq2struct.datasets.spider_lib.preprocess.get_tables import dump_db_json_schema
from seq2struct.datasets.spider import load_tables_from_schema_dict

db_id = "Customers"
my_schema = dump_db_json_schema("data/sqlite_files/{db_id}/{db_id}.sqlite".format(db_id=db_id), db_id)
print(my_schema)

# If you want to change your schema name, then run this; Otherwise you can skip this.
#refine_schema_names(my_schema)

schema, eval_foreign_key_maps = load_tables_from_schema_dict(my_schema)
schema.keys()

dataset = registry.construct('dataset_infer', {
   "name": "spider", "schemas": schema, "eval_foreign_key_maps": eval_foreign_key_maps,
    "db_path": "data/sqlite_files/"
})

for _, schema in dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)

spider_schema = dataset.schemas[db_id]

def db_connect(text):
    try:
        print("okay")
        connection = psycopg2.connect(user="pgadmin",
                                          password="gqfxQz1AiAm",
                                          host="129.146.178.63",
                                          port="1555",
                                          database="airbyte")

        cursor = connection.cursor()
        query = text
        cursor.execute(query)
        res = cursor.fetchall()
        print(res)
        print(type(res))
        df = pd.DataFrame(res, columns=['Product', 'Count'])
        print(df)
        df.to_csv('result.csv')
        #print("Result is:" f'{res[0][0]}')
        print("Executed")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)


    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

app = FastAPI()
app.mount(
    "/templates",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "rat-sql-gap/templates"),
    name="static"
)
templates = Jinja2Templates(directory="./templates")

@app.get("/")
def main(request: Request):
    #languages = ["What is the average price of the product", "What is the phone number of the customer", "What is total quantity of the product", "What is the address of the Kerrin Jambrozek", "What are all the product prices above 200", "Pradeep", "Anish", "Raviteja", "Prameela", "Bhargav"]
        languages = ["What is the total revenue?","What is the average order value?", "What is the average price of the variants?", "What is the total quantity of items?", "What are the unique titles of products", "what is total discount of line items?", "What is the lowest total price in line items", "How many unique cities customers are available?", "How many unique customers are there?", "What is the total price of variants?", "What are all variant titles price below 1000", "What are all unique titles in the smart collections", "What is the currency of country name Sweden", "What are all shipping zones available", "What are the unique names of shipping zones", "who is the vendor? ", "What is the highest tax percentage and name of the province"]
    #languages = ["What are the most common reasons for abandoned carts on our store?"]
    #languages = ["What are the 5 top-selling products on our store?"] #, "What is the average order value on our store?", "What are the most popular payment methods used by our customers?", "What is the sales trend for our store over the last year?", "What are the most common search terms used on our store?"]
    #languages = ["What are the total revenue and month for each order, grouped by month?", "What is the avergae sales in the year 2022"]
    return templates.TemplateResponse("index.html", {"languages": languages, "request": request})

# Define a route for your Dash app
dash_app1 = app_layout(requests_pathname_prefix="/dash/")
app.mount("/dash", WSGIMiddleware(dash_app1.server))

class SelectedValue(BaseModel):
    selected_value: str

def infer(question):
    data_item = SpiderItem(
    text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
    code=None,
    schema=spider_schema,
    orig_schema=spider_schema.orig,
    orig={"question": question}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        output = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
        return output[0]["inferred_code"]

@app.post("/process_selected_value")
def process_selected_value(selected_value: SelectedValue):
    # Retrieve the selected value from the request body
    selected_value = selected_value.selected_value
    # Do something with the selected value, like print it to the console
    print("Selected value:", selected_value)
    # Return a response to the client
    #return {"message": "Selected value processed successfully"}
    n = 1
    q = selected_value
    search_ngrams = [' '.join(q.split()[i:i + n]) for i in range(len(q.split()) - n + 1)]
    print(search_ngrams)
    code = infer(q)
    j = "'terminal'"
    for i in search_ngrams:
        i = f"'{i}'"
        #print(i)
        for d in i.split():
            print("in if")
            d = d[0:3]
            print(d)
            #print(type(d))
            if d.isdigit():
                print('ok')
                code = re.sub(i, d, code)
                print(f'the code is: {code}')
                j = i
                print(f'in if: {j}')
            else:
                code = re.sub(j, i, code)
                j = i
                print(f'in else: {j}')
            print(q)
            #code = "SELECT product_name, Count(*) as total_sales FROM Products_1 GROUP BY product_name ORDER BY total_sales DESC LIMIT 10"
            print(code)
    db_connect(code)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)