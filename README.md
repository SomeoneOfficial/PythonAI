# SPY Stock Prediction with TensorFlow.js

This project is a simple web application that predicts the next day's closing price of the SPY ETF using a pre-trained LSTM model that runs directly in your browser with TensorFlow.js.

**Live Demo:** This project is designed to be hosted on GitHub Pages. Once pushed to a GitHub repository, you can enable GitHub Pages in the repository settings to see it live.

## How It Works

The application uses a Long Short-Term Memory (LSTM) neural network, a type of model well-suited for time-series data like stock prices.

1.  **Model Training (Offline):** An LSTM model was trained in Python using TensorFlow/Keras on historical SPY data. This is a one-time, offline process. The final trained model (`spy_lstm_model.h5`) and the data scaler (`spy_scaler.gz`) are included in this repository for reference.
2.  **Model Conversion:** The trained Keras model was converted into the TensorFlow.js format (`tfjs_model/model.json` and weight files). This allows the model to be loaded and executed by JavaScript.
3.  **Client-Side Prediction:** When you open the `index.html` page:
    *   Your browser fetches the latest ~100 days of SPY stock data from the Alpha Vantage API.
    *   JavaScript code in `predict.js` preprocesses this data (scaling and sequencing) to match the format the model expects.
    *   TensorFlow.js loads the converted model and runs the prediction on your machine.
    *   The result is displayed on the page.

## How to Use

Simply open the `index.html` file in a web browser, or host the entire repository on a static web server or GitHub Pages.

### API Key Note

This demo uses a placeholder `'demo'` API key for Alpha Vantage, which is heavily rate-limited. For more reliable use, you should get your own free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and replace the placeholder in `predict.js`:

```javascript
// In predict.js, line 29
const apiKey = 'YOUR_API_KEY_HERE';
```

## How to Retrain the Model

If you wish to retrain the model with the latest data, you will need to set up a local Python environment.

1.  **Prerequisites:**
    *   Python 3.x
    *   Pip

2.  **Setup:**
    *   Clone the repository.
    *   You will need to temporarily restore the Python files for training. You can do this via git history or by re-creating them. The necessary files are `data_handler.py` and `model.py` (their content is in the commit history).
    *   Install the required Python libraries:
        ```bash
        pip install -r requirements.txt
        pip install tensorflowjs
        ```

3.  **Train the Model:**
    *   Run the training script. This will generate `spy_lstm_model.h5` and `spy_scaler.gz`.
        ```bash
        python model.py
        ```

4.  **Convert the New Model:**
    *   Convert the newly trained `.h5` model to the TensorFlow.js format.
        ```bash
        tensorflowjs_converter --input_format=keras spy_lstm_model.h5 tfjs_model/
        ```

5.  **Update Scaler Parameters:**
    *   Run the `extract_scaler.py` script (also available in git history) to get the new `min` and `scale` values.
        ```bash
        python extract_scaler.py
        ```
    *   Copy the output and update the `SCALER_MIN` and `SCALER_SCALE` constants in `predict.js`.

## Disclaimer

This project is for educational purposes only. Stock market prediction is inherently unreliable, and this model should **NOT** be used for making any financial decisions.
