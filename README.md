# 📊 Stock Price Prediction App

A simple and interactive **Stock Price Prediction Web Application** built with **Streamlit**, **yfinance**, **Linear Regression**, and **Artificial Neural Network (ANN)** models.  
Users can fetch stock data, explore charts, and generate predictions easily.

---

## 🚀 Features

### 🏠 Home Page
- Welcome screen  
- Overview of how the app works  
- Sidebar navigation  

### 📈 Dashboard Page
- Historical stock price chart  
- SMA50 & SMA200 moving averages  
- Volume chart  
- Future prediction visualization  

### 🔮 Prediction Page
- Enter stock ticker  
- Select date range  
- Choose ML model (Linear Regression / ANN)  
- Predict stock closing price  
- Actual vs Predicted chart  
- Download prediction CSV  

---

## 🧠 Machine Learning Models Used

- **Linear Regression Model** (`linear_regression.joblib`)
- **ANN Model** (`ann_model.h5`)
- **MinMaxScaler** (`minmax_scaler.joblib`)

**Features used:**
- Open  
- High  
- Low  
- Close  
- Volume  

---

## 📦 Project Structure
stock-prediction-app/
│── app.py
│── requirements.txt
│── README.md
│── models/
│ ├── linear_regression.joblib
│ ├── ann_model.h5
│ ├── minmax_scaler.joblib
│── notebook/
│ ├── Internship_Yfinance
│── .gitignore


---

## 🛠 Installation Instructions

### 1️⃣ Clone the repository

git clone https://github.com/Sakshi-Srivastava19/stock-prediction-app.git

cd stock-prediction-app

### 2️⃣ Create virtual environment

python -m venv venv

venv\Scripts\activate

### 3️⃣ Install dependencies

pip install -r requirements.txt

### ▶️ Run the Streamlit App
streamlit run app.py

### 🌐 Deployment Options

Streamlit Cloud

### Run the app: https://stock-prediction-app1922.streamlit.app/ 

## 🙌 Author

## Sakshi Srivastava
**AI | ML | Data Science Enthusiast**

⭐ If you like this project, please give it a star on GitHub!

---

If you want, I can also provide:

✅ `requirements.txt`  
✅ GitHub badges  
✅ Project screenshots section  
✅ Demo GIF  

Just tell me!

