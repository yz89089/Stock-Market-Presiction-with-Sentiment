# Stock Market Prediction with News Sentiment

This project explores how financial news sentiment can improve stock market movement prediction, using machine learning models such as Random Forest.

## Project Structure
```
project/  
├── data/                # Datasets and processed CSV files  
├── notebooks/           # Jupyter Notebooks (EDA, modeling, etc.)   
├── requirements.txt     # Required Python packages  
└── README.md            # Project overview and guide  
```
## Aditional Libraries
You can also get the full list of python dependencies and versions in `requirements.txt`
```
pandas
numpy
matplotlib
scikit-learn
torch
transformers
vaderSentiment
yfinance
newsapi-python
seaborn
statsmodels
python-dotenv
openai
```
## Project Highlights

- Collected and cleaned financial headlines from CNBC, Guardian, and Reuters
- Extracted sentiment scores using VADER and FinBERT
- Engineered technical and temporal features from SPY price data (2018–2020)
- Built Random Forest classifiers to predict 3-class SPY movement (Up, Down, Neutral)
- Simulated trading based on predictions vs. monkey/random and buy-and-hold strategies

## Execution Instructions

### 1. Clone the Repository
```
bash
git clone https://github.sfu.ca/xya134/curiousminds.git
cd curiousminds
```
### 2. Setup Virtual Environment
```
python3 -m venv finance_env
source finance_env/bin/activate
pip install -r requirements.txt
```
### 3. Run Jupyter Notebooks
```
jupyter notebook
```
### 4. Recommended Order of Execution
1. **`notebooks/01_prepare_datasets.ipynb`**  
   This notebook does the following things:
   * read news headlines from three headline sources, pre-process, clean, and integrate all headliens into a well structured csv file 'data/cleaned_all_headlines.csv'
   * retrieve the stock market information using `yfinance` according to the start date and end date of the headline dataset. Extract basic stock features such as spy closing price, etc.
   * generate more market related techincal feature such as rolling mean, validity, etc.
  
2. **`notebooks/02_sentiment_analysis.ipynb`**  
   This notebook does the following things:
   * apply sentiment analysis on cleaned news headlines using VADER, FinBERT, OpenAI, and LM methods
   * assign sentiment scores and sentiment labels to each headline
   * aggregate sentiment information by date to generate daily sentiment score and daily sentiment label, and save to CSV files (such as 'data/daily_sentiment_finbert.csv') for further use

3. **`notebooks/03_merge_datasets.ipynb`**  
   This notebook does the following things:
   * merge the daily sentiment dataset with SPY stock market features on the date column
   * prepare the final dataset for model training by generating target variables and spliting to training/testing datasets, and save to 'data/train_dataset.csv' and 'data/test_dataset.csv'

4. **`notebooks/05_rf_model.ipynb`**  
   This notebook does the following things:
   * train Random Forest models to predict 3-class SPY market direction (down, neutral, up)
   * experiment with different sets of features: basic, technical, and sentiment
   * save prediction results ('data/test_result_rf.csv') for comparison and visualization

5. **`notebooks/07_compare_results.ipynb`**  
   This notebook does the following things:
   * compare the predictions of different models using multiple metrics (accuracy, directional hit rate, precision, recall, F1)
   * generate comparison bar charts to visualize model performance across feature sets
   * design and compute a custom metric focusing on directional prediction accuracy (ignoring neutral)
   * create a simulation game to compare buy/sell/hold strategy by our model vs random decisions
