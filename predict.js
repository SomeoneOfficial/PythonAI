// --- Configuration ---
const MODEL_PATH = './tfjs_model/model.json';
const SEQUENCE_LENGTH = 60;
// These parameters are extracted from the Python scaler and must be hardcoded.
const SCALER_MIN = -0.1397788153496879;
const SCALER_SCALE = 0.0017965682766192096;

// --- DOM Elements ---
const statusText = document.getElementById('status-text');
const loader = document.getElementById('loader');
const predictionBox = document.getElementById('prediction-box');
const predictionResult = document.getElementById('prediction-result');
const errorBox = document.getElementById('error-box');

// --- Helper Functions ---

/**
 * Scales a single value using the pre-calculated min and scale.
 * Formula: (value - min) * scale
 * @param {number} value The raw stock price.
 * @returns {number} The scaled stock price.
 */
function scaleValue(value) {
    return (value - SCALER_MIN) * SCALER_SCALE;
}

/**
 * Inverse transforms a single scaled value back to its original price.
 * Formula: (scaled_value / scale) + min
 * @param {number} value The scaled prediction value.
 * @returns {number} The actual predicted stock price.
 */
function inverseTransformValue(value) {
    return (value / SCALER_SCALE) + SCALER_MIN;
}


/**
 * Fetches the last ~100 days of SPY data from Alpha Vantage.
 * NOTE: This uses a demo API key. It is rate-limited and should be replaced
 * with a personal key for any real use.
 * @returns {Promise<Array<number>>} A promise that resolves to an array of closing prices.
 */
async function fetchStockData() {
    statusText.textContent = 'Fetching latest stock data...';
    // Using a demo key for this educational project.
    // It's recommended to get a free key from https://www.alphavantage.co/support/#api-key
    const apiKey = 'demo';
    const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&outputsize=compact&apikey=${apiKey}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`API request failed with status: ${response.status}`);
        }
        const data = await response.json();
        if (data['Error Message'] || !data['Time Series (Daily)']) {
            throw new Error(data['Note'] || 'Invalid API response from Alpha Vantage. This may be due to API rate limits with the demo key.');
        }

        const timeSeries = data['Time Series (Daily)'];
        // The API returns data in reverse chronological order, so we need to sort it.
        const dates = Object.keys(timeSeries).sort((a, b) => new Date(a) - new Date(b));
        const closingPrices = dates.map(date => parseFloat(timeSeries[date]['4. close']));

        return closingPrices;
    } catch (error) {
        throw new Error(`Failed to fetch or parse stock data: ${error.message}`);
    }
}


// --- Main Execution Logic ---

/**
 * Main function to run the entire prediction process.
 */
async function main() {
    try {
        // 1. Load the TensorFlow.js model
        const model = await tf.loadLayersModel(MODEL_PATH);
        statusText.textContent = 'Model loaded successfully.';

        // 2. Fetch and process data
        const closingPrices = await fetchStockData();
        if (closingPrices.length < SEQUENCE_LENGTH) {
            throw new Error(`Not enough data for prediction. Needed ${SEQUENCE_LENGTH}, got ${closingPrices.length}.`);
        }

        statusText.textContent = 'Processing data and making prediction...';

        // 3. Get the last `SEQUENCE_LENGTH` prices and scale them
        const recentData = closingPrices.slice(-SEQUENCE_LENGTH);
        const scaledData = recentData.map(price => scaleValue(price));

        // 4. Create the input tensor for the model
        // Shape should be [1, sequence_length, 1]
        const inputTensor = tf.tensor(scaledData).reshape([1, SEQUENCE_LENGTH, 1]);

        // 5. Make the prediction
        const predictionTensor = model.predict(inputTensor);
        const predictionScaled = await predictionTensor.data();

        // 6. Inverse transform the prediction and display it
        const predictionActual = inverseTransformValue(predictionScaled[0]);

        statusText.style.display = 'none';
        loader.style.display = 'none';
        predictionResult.textContent = `$${predictionActual.toFixed(2)}`;
        predictionBox.style.display = 'block';

    } catch (error) {
        console.error(error);
        statusText.style.display = 'none';
        loader.style.display = 'none';
        errorBox.textContent = `Error: ${error.message}`;
    }
}

// Run the main function when the script loads
main();
