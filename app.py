from flask import Flask, render_template
import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import requests

app = Flask(__name__)

# Tải mô hình ARIMA
model = joblib.load('arima_model.joblib')

def get_covid_data():
    """Lấy dữ liệu COVID-19 từ API."""
    url = "https://disease.sh/v3/covid-19/historical/VietNam?lastdays=all"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        timeline_data = data['timeline']
        df = pd.DataFrame(timeline_data)
        df = df.reset_index().rename(
            columns={
                'index': 'date',
                'cases': 'cases',
                'deaths': 'deaths',
                'recovered': 'recovered'
            })
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        covid_data = df.set_index('date')
        covid_data = covid_data[['cases']]
        covid_data = covid_data.reset_index()
        return covid_data
    else:
        raise Exception("Lỗi khi lấy dữ liệu từ API")

def predict_covid():
    """Dự đoán số ca nhiễm và xu hướng cho ngày mai."""
    covid_data = get_covid_data()
    
    # Xử lý dữ liệu
    if covid_data.isnull().values.any():
        covid_data.fillna(method='ffill', inplace=True)

    # Chuẩn bị dữ liệu cho mô hình ARIMA
    covid_data = covid_data.sort_values(by=['date'])
    covid_data = covid_data.set_index('date')
    covid_data = covid_data[['cases']]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_size = int(len(covid_data) * 0.8)
    train_data = covid_data[:train_size]

    # Dự đoán số ca nhiễm cho ngày mai
    predictions = model.predict(start=len(train_data), end=len(train_data))

    # Xác định xu hướng tăng hay giảm
    previous_cases = train_data['cases'][-1]
    if predictions[0] > previous_cases:
        trend = "Tăng"
    elif predictions[0] < previous_cases:
        trend = "Giảm"
    else:
        trend = "Không đổi"

    return int(predictions[0]), trend

@app.route('/')
def index():
    """Hiển thị trang chủ."""
    prediction, trend = predict_covid()
    return render_template('index.html', prediction=prediction, trend=trend)

if __name__ == '__main__':
    app.run(debug=True)