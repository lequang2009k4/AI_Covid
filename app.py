from flask import Flask, render_template
import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import requests
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
# Tải mô hình ARIMA
model = joblib.load('./arima_model.joblib')

def get_covid_data():
    url = "http://14.225.218.213:1337/api/statistics?pagination[page]=1&pagination[pageSize]=100&sort[0]=date:desc"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['data'])

        # Chuyển đổi kiểu dữ liệu của cột 'date' thành datetime và đặt làm index
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.set_index('date')
        # Lọc dữ liệu và chuẩn bị cho mô hình ARIMA
        covid_data = df[['cases']]

        # Xử lý dữ liệu bị thiếu (nếu có)
        if covid_data.isnull().values.any():
            covid_data.fillna(method='ffill', inplace=True)

        # Sắp xếp dữ liệu theo ngày tháng
        covid_data = covid_data.sort_index()
        # Chuyển đổi cột 'cases' sang kiểu dữ liệu số
        covid_data['cases'] = pd.to_numeric(covid_data['cases'], errors='coerce')

        # Xử lý các giá trị không hợp lệ (ví dụ: thay thế bằng giá trị trung bình)
        covid_data['cases'].fillna(covid_data['cases'].mean(), inplace=True)
        return covid_data
    else:
        raise Exception("Lỗi khi lấy dữ liệu từ API")

def predict_covid():
    """Dự đoán số ca nhiễm và xu hướng cho ngày mai."""
    covid_data = get_covid_data()
    predictions = model.predict(start=len(covid_data), end=len(covid_data))
    predicted_value = predictions.iloc[0]
    previous_cases = covid_data['cases'].iloc[-1]

    # Xác định xu hướng tăng hay giảm

    if predicted_value > previous_cases:
        trend = "Tăng"
    elif predicted_value < previous_cases:
        trend = "Giảm"
    else:
        trend = "Không đổi"
    return int(predicted_value), trend

@app.route('/')
def index():
    """Hiển thị trang chủ."""
    prediction, trend = predict_covid()
    return render_template('index.html', prediction=prediction, trend=trend)
class CovidPrediction(Resource):
    def get(self):
        """Trả về dự đoán số ca nhiễm và xu hướng."""
        try:
            prediction, trend = predict_covid()
            return {
                'prediction': prediction,
                'trend': trend
            }
        except Exception as e:
            return {'error': str(e)}, 500

api.add_resource(CovidPrediction, '/api/predict')
if __name__ == '__main__':
    app.run(debug=True)