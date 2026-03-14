# Fake News Detection 📰

A machine learning-based web application to detect fake news using Python, Scikit-learn, and Streamlit. This project uses a Logistic Regression model trained on a dataset of real and fake news articles to classify input text.

## 🚀 Features

- **Real-time Prediction:** Enter any news text and get instant results.
- **Machine Learning Powered:** Uses TF-IDF Vectorization and Logistic Regression for classification.
- **User-Friendly Interface:** Built with Streamlit for a clean and interactive experience.
- **Model Training Script:** Includes a script to retrain the model on updated datasets.

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Machine Learning:** Scikit-learn, Pandas
- **Web Framework:** Streamlit
- **Model Persistence:** Joblib

## 📁 Project Structure

```text
Fake_News_Detection/
├── app.py              # Streamlit web application
├── train_model.py      # Script to train and save the model
├── fake_news_model.pkl # Trained Logistic Regression model
├── vectorizer.pkl      # Saved TF-IDF Vectorizer
├── Fake.csv            # Dataset containing fake news
├── True.csv            # Dataset containing real news
└── README.md           # Project documentation
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Fake_News_Detection
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn streamlit joblib
   ```

## 🚀 Usage

### 1. Training the Model
If you want to retrain the model or if the `.pkl` files are missing, run:
```bash
python train_model.py
```
This will process `Fake.csv` and `True.csv` and generate `fake_news_model.pkl` and `vectorizer.pkl`.

### 2. Running the Web App
To start the Streamlit application, run:
```bash
streamlit run app.py
```
After running, open your browser to the URL provided (usually `http://localhost:8501`).

## 📊 Dataset
The model is trained on a dataset containing thousands of news articles labeled as Real or Fake.
- **Labels:** `0` for Fake, `1` for Real.
- **Columns used:** `text` (The content of the news article).

## 📝 License
This project is for educational purposes. Feel free to use and modify it!
