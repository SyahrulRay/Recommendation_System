from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("cleaned_car_dataset.csv")

# Fitur yang digunakan
features = ['fuel_cleaned', 'car_type_cleaned', 'transmission_cleaned']

df['best_price'] = df['best_price']/100

# Price bucket helper
def price_bucket(price):
    if price < 500:
        return 'very low'
    elif price < 1500:
        return 'low'
    elif price < 2000:
        return 'Medium'
    elif price < 3000:
        return 'high'
    else:
        return 'very high'

df['price_bucket'] = df['best_price'].apply(price_bucket)
features.append('price_bucket')

# Fit encoder
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[features])

# Flask app
app = Flask(__name__)


@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.json

    # Extract and validate input
    try:
        user_data = pd.DataFrame([{
            'fuel_cleaned': input_data['fuel_type'],
            'car_type_cleaned': input_data['car_type'],
            'transmission_cleaned': input_data['transmission'],
            'price_bucket': price_bucket(input_data['monthly_budget'])
        }])
    except KeyError:
        return jsonify({"error": "Missing required input fields"}), 400

    user_vector = encoder.transform(user_data)
    similarities = cosine_similarity(user_vector, encoded_features).flatten()
    df['similarity'] = similarities
    top_cars = df.sort_values(by='similarity', ascending=False).head(10)

    result = top_cars[['id','version', 'best_price', 'fuel_cleaned', 'car_type_cleaned', 'transmission_cleaned']].to_dict(
        orient='records')
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)