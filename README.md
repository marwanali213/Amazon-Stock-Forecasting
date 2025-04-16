# Amazon Stock Forecasting

This project focuses on time series forecasting of Amazon's stock prices for 2025 using ARIMA, machine learning, and deep learning models. It includes data exploration, preprocessing, model evaluation, and future price prediction.

## Project Structure

```
├── .gitignore
├── README.md
├── data/
│   ├── raw/
│   │   └── AMZN_stock_data.csv
│   ├── preprocessed/
│       └── AMZN_stock_data_processed.csv
├── notebooks/
│   ├── data_exploration_and_preprocessing.ipynb
│   ├── arima_model.ipynb
│   ├── ML_models_forecasting.ipynb
│   └── DL-models-forecast.ipynb
├── outputs/
│   ├── figures/
│   │   ├── amazon_stock_closing_price_over_time.png
│   │   ├── amazon_stock_moving_averages.png
│   │   ├── amazon_stock_volume_over_time.png
│   │   ├── correlation_heatmap.png
│   │   ├── daily_price_change_distribution.png
│   │   └── ...
│   └── predictions/
│       └── ...
```

## Notebooks Overview

1. **Data Exploration and Preprocessing**:
   - Analyzes raw stock data for trends, correlations, and distributions.
   - Generates visualizations such as moving averages, volume trends, and correlation heatmaps.
   - Saves preprocessed data to `data/preprocessed/AMZN_stock_data_processed.csv`.

2. **ARIMA Model**:
   - Implements ARIMA for univariate time series forecasting.
   - Predicts Amazon's stock prices for the next 90 days.
   - Saves predictions to `outputs/predictions/arima_forecast_predictions_3months.json`.

3. **Machine Learning Models**:
   - Trains models like Linear Regression, Random Forest, and Gradient Boosting on stock data.
   - Evaluates models using metrics like R² Score and RMSE.
   - Visualizes model performance comparisons.

4. **Deep Learning Models**:
   - Implements CNN-LSTM models for multivariate time series forecasting.
   - Trains models on features like Open, High, Low, Volume, etc.
   - Visualizes training loss and predictions.

## Key Features

- **Data Visualization**:
  - Stock trends, moving averages, and volume analysis.
  - Correlation heatmaps and daily price change distributions.

- **Forecasting Models**:
  - ARIMA for univariate forecasting.
  - Machine learning models for multivariate forecasting.
  - Deep learning models for advanced time series analysis.

- **Outputs**:
  - Figures saved in `outputs/figures/`.
  - Predictions saved in `outputs/predictions/`.

## How to Run

1. **Setup Environment**:
   - Install required Python packages using `pip install -r requirements.txt` (if available).

2. **Run Notebooks**:
   - Start with `data_exploration_and_preprocessing.ipynb` to preprocess data.
   - Use `arima_model.ipynb`, `ML_models_forecasting.ipynb`, and `DL-models-forecast.ipynb` for forecasting.

3. **View Outputs**:
   - Check visualizations in `outputs/figures/`.
   - Review predictions in `outputs/predictions/`.

## Dependencies

- Python 3.10+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `tensorflow`

## Results

- **ARIMA**: Provides a 90-day forecast of closing prices.
- **Machine Learning**: Compares multiple models for accuracy and error metrics.
- **Deep Learning**: Leverages CNN-LSTM for multivariate forecasting.

## Future Work

- Incorporate additional features like macroeconomic indicators.
- Automate hyperparameter tuning for all models.

```