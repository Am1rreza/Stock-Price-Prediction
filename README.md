# 📈 Stock Price Prediction with LSTM

This project demonstrates how to predict stock prices using **Long Short-Term Memory (LSTM)** neural networks in TensorFlow/Keras.  
It focuses on Apple (AAPL) stock data between 2013 and 2018, applying data preprocessing, visualization, model training, and performance evaluation.  

## 📊 Dataset

The project uses the **All Stocks 5-Year Dataset**, which contains daily stock prices (open, high, low, close, volume) for multiple companies between **2013 and 2018**.

- **Shape:** 619,040 rows × 7 columns  
- **Columns:** `date`, `open`, `high`, `low`, `close`, `volume`, `Name`  
- **Companies:** Includes AAPL, AMD, FB, GOOGL, AMZN, NVDA, EBAY, CSCO, IBM, and more.

The `date` column was converted to `datetime` format for time-series analysis.  
For this project, we mainly focus on **Apple (AAPL)** stock prices.

## 🔍 Exploratory Data Analysis (EDA)

Before building the model, we explored the dataset to understand trends and patterns:

- **Stock Prices Over Time:**  
  Plotted the closing prices of selected companies (AAPL, AMD, FB, GOOGL, AMZN, NVDA, EBAY, CSCO, IBM) to visualize historical trends.

- **Trading Volume Analysis:**  
  Plotted trading volumes for the same companies to identify periods of high activity.

- **Apple Stock Focus:**  
  For Apple (AAPL), we analyzed stock prices from **2013 to 2018** and selected a subset for model training.

## 🛠 Data Preprocessing

To prepare the data for the LSTM model, the following steps were performed:

1. **Selecting Closing Prices:**  
   Focused on Apple's `close` prices as the target variable for prediction.

2. **Scaling:**  
   Applied **MinMaxScaler** to normalize the data to the range `[0, 1]`.

3. **Creating Training Data Sequences:**  
   Used the first 95% of the dataset for training. Each training sample consists of a sequence of 60 previous days' prices used to predict the next day's price.

4. **Resulting Input Shape:**  
   The final input for the LSTM model is a 3D array with shape `(number_of_samples, 60, 1)`.

## 🏗 Building the LSTM Model

The project uses a **Long Short-Term Memory (LSTM)** neural network to predict stock prices.  
The architecture consists of:

- **Input Layer:** Matches the shape of the training sequences `(60, 1)`.
- **First LSTM Layer:** 64 units, returns sequences to the next LSTM layer.
- **Second LSTM Layer:** 64 units, processes the output of the first LSTM layer.
- **Dense Layer:** 32 units, adds a fully connected layer for further processing.
- **Dropout Layer:** 50% dropout to reduce overfitting.
- **Output Layer:** Single neuron predicting the next day's stock price.

The model is designed to learn temporal dependencies in historical stock prices to forecast future prices.

## ⚙️ Model Training & Compilation

The LSTM model was compiled and trained using the following setup:

- **Optimizer:** Adam, for adaptive learning rate optimization.
- **Loss Function:** Mean Squared Error (MSE), to measure prediction error.
- **Training Epochs:** 10 epochs to fit the model on the training data.
- **Training Process:** The model learns patterns in sequences of 60 days of historical stock prices to predict the next day's price.

After training, the model's performance is evaluated on unseen test data using **MSE** and **RMSE** metrics.

## 🧪 Preparing Test Data & Making Predictions

To evaluate the model, the last 5% of the dataset was used as test data.  

- **Test Data Preparation:** Sequences of 60 previous days were created from the scaled data for the test period.  
- **Predictions:** The trained LSTM model predicts the next day's stock price for each test sequence.  
- **Inverse Scaling:** Predicted values are transformed back to the original price scale for comparison with actual stock prices.  
- **Performance Metrics:** The model achieved:
  - **Mean Squared Error (MSE):** 26.34
  - **Root Mean Squared Error (RMSE):** 5.13

## 📊 Visualization of Results

To understand the model's performance, we visualize the predicted vs actual stock prices:

- **Training vs Test Data:**  
  Plotted the historical closing prices used for training alongside the actual prices from the test set.  

- **Predictions:**  
  The model's predicted prices are overlaid on the test data to evaluate how closely predictions follow the real stock prices.  

This visualization helps to quickly identify trends, accuracy, and areas where the model may underperform.

## 📝 Conclusion & Insights

- The LSTM model is effective in capturing temporal patterns in stock prices and provides reasonable predictions for short-term forecasting.
- While the model performs well on historical data, stock prices are influenced by many external factors, so predictions should be interpreted cautiously.
- Feature scaling and sequence creation were essential steps to prepare the data for LSTM.
- Visualizations indicate that the model follows the general trend of Apple stock prices, though some deviations exist during high volatility periods.
