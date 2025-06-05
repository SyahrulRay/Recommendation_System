from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data.csv')  # Ganti dengan path datasetmu
df['combined_text'] = df['car_type_cleaned'].astype(str) + ' ' + df['transmission_cleaned'].astype(str) + ' ' + df['fuel_cleaned'].astype(str)

# Vectorizer dan cosine similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    car_type = data.get('car_type')
    transmission = data.get('transmission')
    fuel_type = data.get('fuel')
    input_price_min = data.get('price_min')
    input_price_max = data.get('price_max')
    top_n = data.get('top_n', 10)

    input_text = f"{car_type} {transmission} {fuel_type}"
    input_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()

    df_temp = df.copy()
    df_temp['similarity'] = similarities
    df_temp = df_temp[
        (df_temp['best_price'] >= input_price_min) & 
        (df_temp['best_price'] <= input_price_max)
    ]
    df_temp = df_temp.sort_values(by=['similarity', 'best_price'], ascending=[False, True])

    results = df_temp[['id', 'version', 'best_price', 'fuel_cleaned', 'car_type_cleaned', 'transmission_cleaned']].head(top_n).to_dict(orient='records')
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
