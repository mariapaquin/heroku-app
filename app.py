from flask import Flask
from flask import Response
import pandas as pd
from sklearn.metrics import *
from sklearn import linear_model
from sklearn import metrics
import numpy as np
import io
import gensim.models.keyedvectors as w
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

data = pd.read_pickle('sentiment.pkl')
   
data.polarity[data.polarity == 4] = 1

train_data = data[:5000]
test_data = data[5000:]

model_onehot = linear_model.LogisticRegression(penalty='l2')
model_w2v = linear_model.LogisticRegression(penalty='l2')

Xtest_onehot = 0
ytest_onehot = 0

Xtest_w2v = 0
ytest_w2v = 0

app = Flask(__name__)

@app.route("/")
def homepage():
    
    return """
        <!DOCTYPE html>
        <head>
            <title>Sentiment Classification</title>
            <link rel="stylesheet" href="http://stash.compjour.org/assets/css/foundation.css">
        </head>
        <body>
            <h1> Welcome! </h1>
            <a href="http://127.0.0.1:5000/onehot">One Hot</a>
            <p></p>
            <a href="http://127.0.0.1:5000/w2v">W2V</a>
        </body>
    """


@app.route("/onehot")
def onehot_page():
    return """
         <a href="http://127.0.0.1:5000/trainonehot">Train One Hot</a>
    """

@app.route("/trainonehot")
def trainonehot():
    global data, train_data, test_data
    global model_onehot, Xtest_onehot, ytest_onehot

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(token_pattern="\\w+", lowercase=True)
    vectorizer.fit(data.tweet)
    
    X = vectorizer.transform(train_data.tweet)
    y = train_data.polarity.as_matrix()
    
    model_onehot.fit(X, y)

    Xtest_onehot = vectorizer.transform(test_data.tweet)
    ytest_onehot = test_data.polarity.as_matrix()

    return """
        <p></p>
         <a href="http://127.0.0.1:5000/testonehot">Test One Hot</a>
    """

@app.route("/testonehot")
def testonehot():
    global model_onehot, Xtest_onehot, ytest_onehot

    return str(accuracy_score(model_onehot.predict(Xtest_onehot), ytest_onehot)) + """
        <p></p>
         <a href="http://127.0.0.1:5000/onehotroc.png">One Hot ROC</a>
    """

# Taken from: (link on trello card)
# https://stackoverflow.com/questions/50728328/python-how-to-show-matplotlib-in-flask
@app.route("/onehotroc.png")
def plot_onehot():
    fig = create_onehot_roc()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_onehot_roc():
    global model_onehot, Xtest_onehot, ytest_onehot

    y_score = model_onehot.decision_function(Xtest_onehot)

    fpr, tpr, thresholds = metrics.roc_curve(ytest_onehot, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    fig = Figure()
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='True positive rate', ylabel='False positive rate',
                   title='ROC curve')

    ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

    ax.legend(loc="lower right")
    
    return fig


@app.route("/w2v")
def w2v_page():
    return """
         <a href="http://127.0.0.1:5000/trainw2v">Train  W2V</a>
    """

@app.route("/trainw2v")
def trainw2v():
    global data, train_data, test_data
    global model_w2v, Xtest_w2v, ytest_w2v

    w2v = w.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

    def sum_w2v_vecs(words):
        vec =  np.sum([w2v[w.lower()] if w.lower() in w2v else np.zeros(300) for w in words], axis=0)
        return vec

    X = train_data.tweet.map(lambda x: sum_w2v_vecs(x.split()))
    X = X.apply(pd.Series)

    y = train_data.polarity.as_matrix()

    model_w2v.fit(X, y)

    Xtest_w2v = test_data.tweet.map(lambda x: sum_w2v_vecs(x.split()))
    ytest_w2v = test_data.polarity.as_matrix()

    Xtest_w2v = Xtest_w2v.apply(pd.Series)

    return """ 
        <p></p>
         <a href="http://127.0.0.1:5000/testw2v">Test W2V</a>
    """

@app.route("/testw2v")
def testw2v():
    global model_w2v, Xtest_w2v, ytest_w2v

    return str(accuracy_score(model_w2v.predict(Xtest_w2v), ytest_w2v)) + """
        <p></p>
         <a href="http://127.0.0.1:5000/w2vroc.png">W2V ROC</a>
    """

# Taken from: (link on trello card)
# https://stackoverflow.com/questions/50728328/python-how-to-show-matplotlib-in-flask
@app.route("/w2vroc.png")
def plot_w2vroc():
    fig = create_w2v_roc()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_w2v_roc():
    global model_w2v, Xtest_w2v, ytest_w2v

    y_score = model_w2v.decision_function(Xtest_w2v)

    fpr, tpr, thresholds = metrics.roc_curve(ytest_w2v, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='True positive rate', ylabel='False positive rate',
                   title='ROC curve')
    
    ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

    ax.legend(loc="lower right")
    
    return fig

if __name__ == '__main__':
        app.run(debug=True)
