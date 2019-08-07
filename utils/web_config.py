import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import flask
import pandas as pd
import time
import os
import pickle

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/hello-world-stock.csv')

app = dash.Dash('app', server=server)

# app.scripts.config.serve_locally = False
# dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

app.layout = html.Div([
    html.H1('Stock Tickers'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Apple', 'value': 'AAPL'},
            {'label': 'Coke', 'value': 'COKE'}
        ],
        value='TSLA'
    ),
    dcc.Graph(id='my-graph')
], className="container")


@app.callback(Output('my-graph', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    log_dict, key_begins = reload_dict("/home/biot/projects/research/genetic_algorithm/logs/pso_elso/log")

    best_fitness = log_dict["iteration_end/best_fitness"]
    global_best_fitness = log_dict["iteration_end/global_best_fitness"]
    x_data = log_dict["iteration_end/iteration"]
    y_data = log_dict[key_begins[0] + "/improvement"]
    return {
        'data': [{
            'x': x_data,
            'y': y_data,
            'line': {
                'width': 3,
                'shape': 'line'
            }
        }],
        'layout': {
            'margin': {
                'l': 30,
                'r': 20,
                'b': 30,
                't': 20
            }
        }
    }


def reload_dict(file_path):
    try:
        with open(file_path, "rb") as log_file:
            logs = pickle.load(log_file)
    except FileNotFoundError:
        logs = []
        print("Wrong file path!")

    log_dict = dict()
    for log in logs:
        for key, inner_dict in log.items():
            for inner_key, item in inner_dict.items():
                if key + '/' + inner_key not in log_dict.keys():
                    log_dict[key + '/' + inner_key] = []
                log_dict[key + '/' + inner_key] += [item]

    keys = list(log_dict.keys())
    key_begins = list(set([key.split('/')[0] for key in keys]))
    key_begins.sort()

    return log_dict, key_begins


if __name__ == '__main__':
    app.run_server()
