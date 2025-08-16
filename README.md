# StockProphet 2.0:

This project is an advanced and significantly enhanced version of the original [StockProphet](https://github.com/Aaryyan777/StockProphet), a time series forecasting project for predicting stock prices.

## From StockProphet to StockProphet 2.0

### The Initial State

The original StockProphet was a solid project that demonstrated the use of Facebook's Prophet library for stock price forecasting. It used a few basic technical indicators and achieved respectable results. However, I saw an opportunity to push the boundaries and see how much I could improve the model's performance.

### The Quest for Improvement

The goal was to enhance the model's predictive power by providing it with more information and by fine-tuning its parameters. I hypothesized that a more sophisticated feature set and a more systematic approach to hyperparameter tuning would lead to a significant improvement in accuracy.

### Enhancement Strategy

I adopted a two-pronged approach to enhance the model:

1.  **Advanced Feature Engineering:** I went beyond the basic features of the original model and incorporated more advanced technical indicators:
    *   **MACD (Moving Average Convergence Divergence):** To capture the trend-following momentum of the stock.
    *   **Bollinger Bands (Upper and Lower):** To provide the model with a measure of market volatility.

2.  **Robust Hyperparameter Tuning:** Instead of relying on hardcoded hyperparameters, I implemented a systematic grid search with cross-validation. This allowed me to find the optimal combination of `changepoint_prior_scale`, `seasonality_prior_scale`, and `holidays_prior_scale`, ensuring the model is perfectly calibrated to the data.

## Key Features of StockProphet 2.0

-   **Advanced Feature Set:** A rich set of technical indicators for a more informed forecast.
-   **Optimized Hyperparameters:** A systematic grid search for a perfectly calibrated model.
-   **Superior Performance:** Significant improvements in all key performance metrics.
-   **Comprehensive Documentation:** A detailed guide to the project and the enhancement process.

## The Results: A Leap in Performance

The enhancements resulted in a dramatic improvement in the model's performance:

| Metric                 | Original Model | Enhanced Model |
| :--------------------- | :------------- | :------------- |
| **MAE**                | 1.78           | 0.60           |
| **RMSE**               | 2.00           | 0.74           |
| **MAPE**               | 1.11%          | 0.37%          |
| **R-squared**          | 0.82           | 0.98           |
| **Directional Accuracy** | 92.72%         | 91.19%         |

## What's New and Improved (The Nitty-Gritty)

### Code Enhancements

-   **New Feature Engineering Functions:**
    -   `calc_macd()`: To calculate the MACD and its signal line.
    -   `calc_bb()`: To calculate the upper and lower Bollinger Bands.
-   **Hyperparameter Tuning Implementation:**
    -   A `param_grid` dictionary was defined to hold the hyperparameter search space.
    -   A loop was implemented to iterate through all parameter combinations and perform cross-validation for each.

### Comparison

1. **Feature Engineering**

Original: Only included SMA_10, SMA_30, and RSI.
2.0: Added ATR, MACD, MACD signal, Bollinger Bands (upper/lower), and lag features (y_lag1, y_lag2, y_lag3).
Comment: More features mean the model can capture richer price dynamics, trends, momentum, and volatility.

2. **Hyperparameter Tuning**

Original: Hardcoded best parameters (changepoint_prior_scale, seasonality_prior_scale) found manually.
2.0: Implements systematic grid search with cross-validation to find the best parameters (changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale).
Comment: Removes guesswork and adapts better to data.

3. **Model Evaluation**

Original: Rolling window CV with MAE, RMSE, MAPE, R², and directional accuracy.
2.0: Keeps rolling window CV but also optimizes parameters before CV.
Comment: This two-phase approach (tuning → CV) ensures you get a well-calibrated and robust model.

4. Prediction Script

Original predict.py: Only used SMA, RSI, and MA to predict.
2.0 predict.py: Uses all advanced features and lagged prices for prediction, ensuring feature consistency between training and inference.
Comment: Matches training features to prediction features.

### What I Kept

I retained the core structure of the original project, including:

-   The use of the Prophet model.
-   The data acquisition and basic preprocessing steps.
-   The use of `joblib` for model persistence.

### What I Removed/Replaced

-   **Hardcoded Hyperparameters:** The hardcoded `best_params` dictionary was replaced with a systematic grid search.

## Tech Stack

-   Python
-   Prophet
-   yfinance
-   pandas
-   scikit-learn
-   Matplotlib
-   Joblib

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/StockProphet-Enhanced.git
    cd StockProphet-Enhanced
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the model:**
    ```bash
    python stock_forecaster.py
    ```
4.  **Make predictions:**
    ```bash
    python predict.py
    ```


## Future Directions

-   **Explore More Features:** Incorporate fundamental data (e.g., P/E ratios, earnings reports) and sentiment analysis of news articles.
-   **Experiment with Other Models:** Test other time series models like LSTMs or Transformers.
-   **Build a Web Application:** Create a user-friendly web interface for interacting with the model.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is an enhanced version of the original [StockProphet](https://github.com/Aaryyan777/StockProphet) by Aaryyan777.
