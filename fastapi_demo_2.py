#Importing the required Libraries
import json
import pandas as pd
#from test_chart import app as dash_app
import requests
from fastapi.middleware.wsgi import WSGIMiddleware
#import plotly.express as px
import _jsonnet
import psycopg2
import mysql.connector
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
        connection = mysql.connector.connect(
                                                host='129.146.35.116',
                                                port="9030",
                                                user='root',
                                                password='gqfxQz1AiAm',
                                                database='dashl_shop_dev_'
                                            )

        cursor = connection.cursor()
        query = text
        cursor.execute(query)
        res = cursor.fetchall()
        print(res)
        if len(res)>=1:
            if len(res[0]) == 1:
                for index in range(len(res)):
                    res[index] = res[index] + (0,)
            #print("Result is:" f'{res[0][0]}')
            #print(type(res))
            df = pd.DataFrame(res, columns=['Column1', 'Column2'])
            if (df['Column2'] == 0).all():
                df = df.drop(df.columns[1], axis=1)
                #df = df.astype(float).round(2)
                #df['Column2'] = df['Column2'].astype(float).round(2)
                #df = df.astype(float).round({'Column1': 2})
            else:
                df['Column1'] = df['Column1'].astype(float).round(2)
                df['Column2'] = df['Column2'].astype(float).round(2)


        elif len(res[0])>1:
            df = pd.DataFrame(res, columns=['Column1', 'Column2'])
            #df['Column1'] = df['Column1'].astype(float).round(2)
            #df['Column2'] = df['Column2'].astype(float).round(2)
            df = df.astype(float).round(2)
            #df = df.astype(float).round({'Column1': 2, 'Column2': 2})
            #df = df.applymap(lambda x: round(x, 2))

        #else:
            #df = str(res[0][0])

        #df.to_csv('result.csv')
        print("Executed")
        #return ("Result is:" f'{df}')
        return df

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
        df = "Error while connecting to PostgreSQL"
        return df


    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

app = FastAPI()

origins = ["*"]

app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

app.mount(
    "/templates",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "rat-sql-gap/templates"),
    name="static"
)
templates = Jinja2Templates(directory="./templates")

#@app.get("/")
#def main(request: Request):
    #languages = ["What is the average price of the product", "What is the phone number of the customer", "What is total quantity of the product", "What is the address of the Kerrin Jambrozek", "What are all the product prices above 200", "Pradeep", "Anish", "Raviteja", "Prameela", "Bhargav"]
    #languages = ["What are all products and price", "What is the average price of the product", "What is the phone numebr of the custoner", "What is the total quantity of product", "What is the address of the Kerrin Jambrozek", "What are all the product prices above 200"]
    #languages = ["What are the most common reasons for abandoned carts on our store?"]
    #languages = ["What are the 5 top-selling products on our store?"] #, "What is the average order value on our store?", "What are the most popular payment methods used by our customers?", "What is the sales trend for our store over the last year?", "What are the most common search terms used on our store?"]
    #languages = ["What are the total revenue and month for each order, grouped by month?", "What is the avergae sales in the year 2022"]
    #return templates.TemplateResponse("index.html", {"languages": languages, "request": request})

@app.get("/")
def index():
    return {'message':'Welcome Ask Haathi'}

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

@app.get("/process_selected_value")
def process_selected_value(selected_value: str):
    # Retrieve the selected value from the request body
    #selected_value = selected_value.selected_value
    # Do something with the selected value, like print it to the console
    print("Selected value:", selected_value)
    # Return a response to the client
    #return {"message": "Selected value processed successfully"}
    quest = ["What is the monthly churn rate",
             "What is Weekly churn rate",
             "What is the churn rate of current week",
             "What is the churn rate of current month",
             "What is the churn rate of current year",
             "How many of these customers have placed more than 2 orders?",
             "What is the total quantity of products sold to customers who placed orders over the value of 30?",
             "What is the average number of products per order for customers who placed orders over 30?",
             "How many customers have placed orders by SKU or product variant?",
             "What is the total revenue generated from customers who placed orders by product variant or sku?",
             "How many customers have placed orders with a specific shipping method?",
             "What is the total revenue generated from customers who placed orders by shipping method?",
             "How many customers have placed orders from a date 2023-01-03 to 2023-05-30",
             "What is the total revenue generated from customers who placed orders from a date 2023-01-18 to 2023-05-15"
             ]
    sql = [
        "SELECT o.month_name, ((c.total_customers - COUNT(DISTINCT o.customer_id)) / c.total_customers) * 100 AS monthly_churn_rate_percentage FROM (SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_name, customer_id, id FROM orders WHERE DATE_FORMAT(created_at, '%Y') = DATE_FORMAT(NOW(), '%Y')) o CROSS JOIN (SELECT COUNT(DISTINCT id) AS total_customers FROM customers) c GROUP BY o.month_name, c.total_customers ORDER BY o.month_name",
        "SELECT CONCAT('week_', o.week_number) AS week_number,((c.total_customers - COUNT(DISTINCT o.customer_id)) / c.total_customers) * 100 AS weekly_churn_rate_percentage FROM (SELECT week(created_at) AS week_number, customer_id, id FROM orders WHERE DATE_FORMAT(created_at, '%Y') = DATE_FORMAT(NOW(), '%Y')) o CROSS JOIN (SELECT COUNT(DISTINCT id) AS total_customers FROM customers) c GROUP BY o.week_number, c.total_customers ORDER BY o.week_number",
        "SELECT CONCAT('week_', o.week_number) AS week_number, ((c.total_customers - COUNT(DISTINCT o.customer_id)) / c.total_customers) * 100 AS weekly_churn_rate_percentage FROM (SELECT week(created_at) AS week_number, customer_id, id FROM orders WHERE DATE_FORMAT(created_at, '%Y') = DATE_FORMAT(NOW(), '%Y') AND WEEK(o.created_at) = WEEK(NOW())) o CROSS JOIN (SELECT COUNT(DISTINCT id) AS total_customers FROM customers) c GROUP BY o.week_number, c.total_customers ORDER BY o.week_number",
        "SELECT o.month_name, ((c.total_customers - COUNT(DISTINCT o.customer_id)) / c.total_customers) * 100 AS monthly_churn_rate_percentage FROM (SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_name, customer_id, id FROM orders WHERE DATE_FORMAT(created_at, '%Y') = DATE_FORMAT(NOW(), '%Y') AND MONTH(o.created_at) = MONTH(NOW())) o CROSS JOIN (SELECT COUNT(DISTINCT id) AS total_customers FROM customers) c GROUP BY o.month_name, c.total_customers ORDER BY o.month_name",
        "SELECT o.year, ((c.total_customers - COUNT(DISTINCT o.customer_id)) / c.total_customers) * 100 AS yearly_churn_rate_percentage FROM (SELECT DATE_FORMAT(created_at, '%Y') AS year, customer_id, id FROM orders WHERE DATE_FORMAT(created_at, '%Y') = DATE_FORMAT(NOW(), '%Y')) o CROSS JOIN (SELECT COUNT(DISTINCT id) AS total_customers FROM customers) c GROUP BY o.year, c.total_customers ORDER BY o.year",
        "SELECT COUNT(DISTINCT customer_id) AS customers_with_more_than_2_orders FROM (SELECT customer_id, COUNT(*) AS order_count FROM orders  GROUP BY customer_id) a WHERE  order_count > 2",
        "SELECT SUM(line_items_quantity) AS total_product_quantity FROM orders o WHERE total_price > 30",
        "SELECT avg(line_items_quantity) AS total_product_quantity FROM orders o WHERE total_price > 30",
        "select line_items_name as sku, count(distinct customer_id) as customers_by_variant from orders o left join line_items li  on o.line_items_id =li.id group by 1 order by 1",
        "select line_items_name as sku, sum(o.total_price ) as revenue_by_variant from orders o  left join line_items li on o.line_items_id =li.id group by 1 order by 1",
        "select shipping_lines_code as shipping_method, count(distinct customer_id) as customers_by_shipping_method from orders o group by 1 order by 1",
        "select shipping_lines_code as shipping_method, sum(o.total_price ) as revenue_by_shipping_method from orders o group by 1 order by 1",
        "SELECT Count(*) FROM orders WHERE orders.created_at >= '2023-01-03' AND orders.created_at <= '2023-05-30'",
        "SELECT Sum(orders.total_price) FROM orders WHERE orders.created_at >= '2023-01-18' AND orders.created_at <= '2023-05-15'"
        ]

    if selected_value.strip() in [item.strip() for item in quest]:
        index = [item.strip() for item in quest].index(selected_value.strip())
        code = sql[index]
        print(code)
    else:
        n = 1
        q = selected_value
        print(q)
        search_ngrams = [' '.join(q.split()[i:i + n]) for i in range(len(q.split()) - n + 1)]
        print(search_ngrams)
        code = infer(q)
        j = "'terminal'"
        for i in search_ngrams:
            i = f"'{i}'"
            #print(i)
            for d in i.split():
                d = d[:3]
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
                #code = ["SELECT to_char(order_date, 'MONTH') AS month_name, (t.total_customers - COUNT(DISTINCT c.customer_id)) * 100.0 / t.total_customers AS monthly_churn_rate_percentage FROM customers_1 c LEFT JOIN orders_1 o ON c.customer_id = o.customer_id CROSS JOIN (SELECT COUNT(DISTINCT customer_id) AS total_customers FROM customers_1) t WHERE EXTRACT(YEAR FROM o.order_date) = EXTRACT(YEAR FROM CURRENT_DATE) AND o.order_date >= DATE_TRUNC('year', CURRENT_DATE) AND o.order_date < DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '1 year' GROUP BY month_name, t.total_customers ORDER BY EXTRACT(MONTH FROM MIN(order_date))", "select 'week_' || extract(week from o.order_date) AS week_number, (t.total_customers - COUNT(DISTINCT c.customer_id)) * 100.0 / t.total_customers AS monthly_churn_rate_percentage from customers_1 c LEFT JOIN orders_1 o ON c.customer_id = o.customer_id CROSS JOIN (SELECT COUNT(DISTINCT customer_id) AS total_customers FROM customers_1) t where EXTRACT(YEAR FROM o.order_date) = EXTRACT(YEAR FROM CURRENT_DATE) AND o.order_date >= DATE_TRUNC('year', CURRENT_DATE) AND o.order_date < DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '1 year' GROUP by week_number, t.total_customers ORDER by SUBSTRING('week_' || extract(week from o.order_date) FROM '\d+')::int ASC"]
                print(code)

    result = db_connect(code)
    print(result)
    #db_connect(code)
    if type(result)==pd.core.frame.DataFrame:
        df_dict = result.to_dict(orient='records')
    else:
        df_dict=result
    #table_html = result.to_html(index=False)
    additional_string = f'{selected_value} : {code}'
    #response_content = f"{additional_string}<br><br>{table_html}"
    response_dict = {'additional_text': additional_string, 'dataframe': df_dict}
    #return HTMLResponse(content=response_content, status_code=200)
    return response_dict

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)