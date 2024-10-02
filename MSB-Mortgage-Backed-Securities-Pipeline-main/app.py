from flask import Flask, render_template, request, url_for
import joblib
import pandas as pd
import numpy as np
from model import CustomPipeline
import sklearn
# print("Current version of scikit-learn", sklearn.__version__)


app = Flask(__name__)

cat_cols = {
    'PropertyState': ['il', 'co', 'ks', 'ca', 'nj', 'wi', 'fl', 'ct', 'ga', 'tx', 'md',
                      'ma', 'sc', 'wy', 'nc', 'az', 'in', 'ms', 'ny', 'wa', 'ar', 'va',
                      'mn', 'la', 'pa', 'or', 'ri', 'ut', 'mi', 'tn', 'al', 'mo', 'ia',
                      'nm', 'nv', 'oh', 'ne', 'vt', 'hi', 'id', 'pr', 'dc', 'gu', 'ky',
                      'nh', 'sd', 'me', 'mt', 'ok', 'wv', 'de', 'nd', 'ak'],
    'PropertyType': ['sf', 'pu', 'co', 'mh', 'cp', 'lh'],
    'Occupancy': ['o', 'i', 's'],
    'FirstTimeHomebuyer_yes': ['n', 'y'],
    'Channel': ['t', 'r', 'c', 'b'],
    'PPM': ['n', 'y'],
    'LoanPurpose': ['p', 'n', 'c'],
}

num_cols = {
    'MonthsDelinquent': 0,
    'CreditScoreRange_good': 1,
    'CreditScoreRange_excellent': 1,
    'CreditScoreRange_fair': 1,
    'CreditScoreRange_poor': 1,
    'LTV_med': 1,
    'FirstPaymentDate': 1,
    'OrigUPB': 100000,
    'OrigInterestRate': 1,
    'OrigLoanTerm': 100,
    'NumBorrowers': 1,
    'MIP': 1,
    'DTI': 0,
    'DTI_fraction': 0,
}


@app.route("/")
def index():
    return render_template("index.html", cat_cols=cat_cols, num_cols=num_cols)


@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        with open('model/combined_pipeline.pkl', 'rb') as f:
            model_pipeline = joblib.load(f)

        data = {}
        for variable in ['CreditScoreRange_good', 'CreditScoreRange_poor', 'CreditScoreRange_excellent', 'CreditScoreRange_fair'
                         'LTV_med', 'FirstTimeHomebuyer_yes',
                         'MIP', 'Occupancy', 'DTI', 'DTI_fraction', 'OrigUPB', 'OrigInterestRate', 'Channel',
                         'PPM', 'PropertyState', 'PropertyType', 'LoanPurpose', 'OrigLoanTerm', 'NumBorrowers', 'MonthsDelinquent'
                         ]:
            data[variable] = request.form.get(variable)

        sample = pd.DataFrame([data])

        # print("Before labeling", sample.info())
        # Label encoding for categoricals
        for colname in cat_cols:
            sample[colname], _ = sample[colname].factorize()
        # print("After labeling", sample.info())

        prediction, classification = model_pipeline.predict(sample)
    return render_template('results.html', prediction_text='La prediction du risk de prepaiement est :{}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
