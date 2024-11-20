from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained ARIMA model
model = joblib.load('arima_sales_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the number of months to forecast
        months = int(request.form['months'])
        
        # Predict sales for the next 'months' months
        forecast = model.forecast(steps=months)
        
        # Prepare the forecasted data
        forecast_dates = pd.date_range(pd.to_datetime('today'), periods=months+1, freq='M')[1:]
        forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Sales Forecast'])
        
        return render_template('index.html', forecast_data=forecast_df.to_html(classes='table table-striped', header=True, index=True))

if __name__ == "__main__":
    app.run(debug=True)
