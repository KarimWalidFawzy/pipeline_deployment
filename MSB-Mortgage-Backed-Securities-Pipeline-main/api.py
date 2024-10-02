from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from model import CustomPipelineWithFeatureSelection

app = Flask(__name__)
# Chargement du modèle
with open('model/pipeline_v2.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)

    # Obtenir les prédictions
    y_class_pred, y_reg_pred = model_pipeline.predict(query_df)

    # Convertir les prédictions en listes
    return jsonify({
        'class_predictions': y_class_pred.tolist(),
        'reg_predictions': y_reg_pred.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
