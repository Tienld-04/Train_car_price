import joblib
import pandas as pd
from example_3 import covert_columns

fake_input = pd.DataFrame([{
    'full_name': 'Hyundai i20 Sport',
    'insurance': 'Comprehensive',
    'transmission_type': 'Manual',
    'owner_type': 'First',
    'fuel_type': 'Petrol',
    'body_type': 'Hatchback',
    'city': 'Bangalore',
    'registered_year': 2018,
    'engine_capacity': '1197cc',
    'kms_driven': '50,000 Kms',
    'max_power': '82 bhp',
    'seats': 5,
    'mileage': '18.6 kmpl',
    'resale_price': '₹5.2 Lakh'
}])

model_path = "D:\\Python\\API_model\\ML3\\car_price\\trained_car_price_model.pkl"
model = joblib.load(model_path)

fake_input_processed = covert_columns(fake_input)
prediction = model.predict (fake_input_processed)

print("📌 Giá trị dự đoán resale_price là: ₹{:,}".format(int(prediction[0])))

