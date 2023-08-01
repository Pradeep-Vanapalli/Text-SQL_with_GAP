from dash_holoniq_apexcharts import DashApexcharts
import dash
from dash import html
import dash_core_components as dcc
import flask
import pandas as pd

#app = dash.Dash(__name__)

df = pd.read_csv('result.csv')
SALES_CHART = {
        "chart": {
            "fontFamily": "Inter, sans-serif",
            "foreColor": "#6B7280",
            "toolbar": {"show": False},
        },
        "fill": {
            "type": "solid",
            "opacity": 0.3,
        },
        "dataLabels": {"enabled": False},
        "tooltip": {
            "style": {
                "fontSize": "14px",
                "fontFamily": "Inter, sans-serif",
            },
        },
        "grid": {
            "show": False,
        },
        "xaxis": {
            "categories": df['Product'],
            "labels": {
                "style": {
                    "colors": ["#6B7280"],
                    "fontSize": "14px",
                    "fontWeight": 500,
                },
            },
            "axisBorder": {
                "color": "#F3F4F6",
            },
            "axisTicks": {
                "color": "#F3F4F6",
            },
        },
        "yaxis": {
            "labels": {
                "style": {
                    "colors": ["#6B7280"],
                    "fontSize": "14px",
                    "fontWeight": 500,
                },
                'formatter': "${value}"
            },
        },
        "responsive": [{"breakpoint": 1024, "options": {"xaxis": {"labels": {"show": False}}}}],
    }


def sales_chart():
    series = {"name": "Revenue", "data": df['Count'], "color": "#0694a2"}
    chart = DashApexcharts(options=SALES_CHART, series=[series], type='area', width=420)

    return chart

def profit_chart():
    series = {"name": "Profit", "data": df['Count'], "color": "#2eb872"}
    chart = DashApexcharts(options=SALES_CHART, series=[series], type='line', width=420)

    return chart

def sca_chart():
    series = {"name": "sca", "data": df['Count'], "color": "#2eb872"}
    chart = DashApexcharts(options=SALES_CHART, series=[series], type='scatter', width=420)

    return chart

def app_layout(requests_pathname_prefix):
    server = flask.Flask(__name__)
    app = dash.Dash(__name__, server=server, requests_pathname_prefix=requests_pathname_prefix)
    app.layout = html.Div([
        dcc.Loading(
            id="loading",
            children=[html.Div([
                html.Div(sales_chart(), className='ten columns'),
                html.Div(profit_chart(), className='ten columns'),
                html.Div(sca_chart(), className='ten columns')
            ])],
            type="default",
        ),
    ])
    return app
