#Importing the required Libraries
import json
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as px
import os
import _jsonnet
import psycopg2
import re
import os
import torch
from seq2struct.commands.infer import Inferer
from seq2struct.datasets.spider import SpiderItem
from seq2struct.utils import registry
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")

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
from seq2struct.utils.api_utils import refine_schema_names

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

def mixed_plots(data):
    df = pd.DataFrame(data, columns=['product', 'price'])

    plot = px.Figure(data=[px.Scatter(
    x=df['product'],
    y=df['price'],
    mode='lines', )
    ])

# Add dropdown
    plot.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=["type", "line"],
                        label="Line Plot",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "bar"],
                        label="Bar Chart",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "histogram"],
                        label="Histogram Plot",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "pie"],
                        label="Pie Plot",
                        method="restyle"
                    )
                ]),
            ),
        ]
    )

    plot.show()

def db_connect(text):
    try:
        print("okay")
        connection = psycopg2.connect(user="pgadmin",
                                          password="gqfxQz1AiAm",
                                          host="129.146.178.63",
                                          port="1555",
                                          database="airbyte")

        cursor = connection.cursor()
        # Print PostgreSQL Connection properties
        #print(connection.get_dsn_parameters(), "\n")

        # Print PostgreSQL version
        #cursor.execute("SELECT version();")

        #record = cursor.fetchone()
        #print("You are connected to - ", record, "\n")

        query = text
        cursor.execute(query)
        res=cursor.fetchall()
        print("Result is:" f'{res[0][0]}')
        #bar_chart(res)
        #line_chart(res)
        #scatter(res)

        print("Executed")

        mixed_plots(res)

        #pie chart
        # df = pd.DataFrame(res, columns=['product', 'price'])
        # df['product_id'] = df['product_id'].astype(str)
        # print(df)
        # fig = px.pie(df, values='price', names='product')
        # fig.show()

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
    name="static",
)
templates = Jinja2Templates(directory="./templates")

@app.get("/")
def main(request: Request):
    #languages = ["What is the average price of the product", "What is the phone number of the customer", "What is total quantity of the product", "What is the address of the Kerrin Jambrozek", "What are all the product prices above 200", "Pradeep", "Anish", "Raviteja", "Prameela", "Bhargav"]
    #languages = ["What are all products and price", "What is the average price of the product", "What is the phone numebr of the custoner", "What is the total quantity of product", "What is the address of the Kerrin Jambrozek", "What are all the product prices above 200"]
    #languages = ["What are the most common reasons for abandoned carts on our store?"]
    languages = ["What are the 5 top-selling products on our store?"] #, "What is the average order value on our store?", "What are the most popular payment methods used by our customers?", "What is the sales trend for our store over the last year?", "What are the most common search terms used on our store?"]
    return templates.TemplateResponse("index.html", {"languages": languages, "request": request})

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
    n = 2
    q = selected_value
    search_ngrams = [' SELECT collection_id, created_at, id, "position", product_id, sort_value, updated_at'.join(q.split()[i:i + n]) for i in range(len(q.split()) - n + 1)]
    print(search_ngrams)
    code = infer(q)
    j = "'terminal'"
    for i in search_ngrams:
        i = f"'{i}'"
        #print(i)
        for d in i.split():
            d = d[0:3]
            #print(d)
            #print(type(d))
            if d.isdigit():
                print('ok')
                code = re.sub(i, d, code)
                j = i
                #print(f'in if: {j}')
            else:
                code = re.sub(j, i, code)
                j = i
                #print(f'in else: {j}')
            print(q)
            #code = "SELECT product_name, Count(*) as total_sales FROM Products_1 GROUP BY product_name ORDER BY total_sales DESC LIMIT 5"
            print(code)
    db_connect(code)

'''def bar_chart(data):
    df = pd.DataFrame(data, columns=['product', 'price'])
    # df['product_id'] = df['product_id'].astype(str)
    print(df)
    fig = px.pie(df, values='price', names='product')
    fig.show()

def line_chart(data):
    df = pd.DataFrame(data, columns=['product', 'price'])
    # df['product_id'] = df['product_id'].astype(str)
    print(df)
    fig = px.bar(df, y='price', x='product')
    fig.show()

def scatter(data):
    df = pd.DataFrame(data, columns=['product', 'price'])
    print(df)
    fig = px.scatter(df, x="price", y="product")
    fig.show()'''

if __name__ == "__main__":
    uvicorn.run(app)