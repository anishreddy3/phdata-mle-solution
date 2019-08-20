import pickle
from flask import Flask, request, jsonify, redirect, url_for, flash
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import traceback

application = Flask(__name__)




@application.route('/api', methods=['POST'])
def api():
    if model:
        try:
            json_ = request.json
            print(json_)
            df = pd.DataFrame.from_dict(json_)

            df = df.reindex(columns=model_col, fill_value=0)
            prediction = list(model.predict_proba(df))

            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'trace' : traceback.format_exc()})

    else:
        print('Please Train the model first')
        return('No model to run')


if __name__ == '__main__':
    try:
        port =int(sys.argv[1])
    except:
        port = 9999
    model = pickle.load(open('latePaymentsModel.pkl','rb'))
    print("model loaded")
    model_col = joblib.load('model_columns.pkl')
    print('model columns loaded')
    application.run(host = '0.0.0.0',port=port, debug=True)
