import json
import plotly
import pandas as pd
import joblib
import sys

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorised_tweets', engine)

# load model
sys.path.append('../models') # allow import of CustomGridSearchCV and tokenize
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_sums = df.iloc[:,4:].sum(axis=0).sort_values()
    category_counts = category_sums.values.tolist()
    category_names = category_sums.index.tolist()

    n_labels = df.iloc[:,4:].sum(axis=1)
    n_label_counts = df.groupby(n_labels).count()['message']
    n_label_names = n_label_counts.index.tolist()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages’ Genres',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=category_counts,
                    y=category_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Count of Messages Per Category',
                'yaxis': {
                    'title': "Category",
                    'automargin': "True"
                },
                'xaxis': {
                    'title': "Message Count"
                },
                'height': "1000",
                'margin': {
                    'l': "200"
                },
            }
        },

        {
            'data': [
                Bar(
                    x=n_label_names,
                    y=n_label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages’ Number of Categories',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Messages’ Number of Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', genre_names=genre_names, ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()