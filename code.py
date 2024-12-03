import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib  # Import thư viện joblib
# Lấy dữ liệu từ API
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
else:
    print("Lỗi khi lấy dữ liệu từ API:", response.status_code)

# Xử lý dữ liệu
covid_data.info()
if covid_data.isnull().values.any():
    covid_data.fillna(method='ffill', inplace=True)

# Chuẩn bị dữ liệu cho mô hình ARIMA
covid_data = covid_data.sort_values(by=['date'])
covid_data = covid_data.set_index('date')
covid_data = covid_data[['cases']]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_size = int(len(covid_data) * 0.8)
train_data = covid_data[:train_size]
test_data = covid_data[train_size:]

# Xây dựng mô hình ARIMA
model = ARIMA(train_data['cases'],
              order=(3, 1, 0),
              seasonal_order=(2, 1, 0, 12))

# Huấn luyện mô hình
model_fit = model.fit()
# Lưu mô hình vào file joblib

filename = 'arima_model.joblib'
joblib.dump(model_fit, filename)
# Dự đoán số ca nhiễm cho ngày mai
predictions = model_fit.predict(start=len(train_data),
                                 end=len(train_data))

# Xác định xu hướng tăng hay giảm
previous_cases = train_data['cases'][-1]
if predictions[0] > previous_cases:
    trend = "Tăng"
elif predictions[0] < previous_cases:
    trend = "Giảm"
else:
    trend = "Không đổi"

# In kết quả dự đoán và xu hướng
print(f"Dự đoán số ca nhiễm ngày mai: {predictions[0]:.0f}")
print(f"Xu hướng: {trend}")