# 🏦 Customer Churn Prediction — ANN Classification

A deep learning web app that predicts whether a bank customer is likely to churn, built with TensorFlow, Scikit-learn, and Streamlit.

---

## 🔗 Live Demo

https://ann-classification-churn-wau5umkv7nkibcngamcwup.streamlit.app/
---

## 📌 Problem Statement

Banks lose significant revenue when customers leave (churn). This app uses a trained Artificial Neural Network to predict the probability of a customer churning based on their profile — enabling early intervention and retention strategies.

---

## 🧠 Model Architecture

| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | Dense | 12 | ReLU |
| Hidden | Dense | 6 | ReLU |
| Output | Dense | 1 | Sigmoid |

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Training Accuracy:** ~88%  
- **Validation Accuracy:** ~85%

---

## 📊 Dataset

**Churn_Modelling.csv** — 10,000 bank customer records

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (1/0) |
| IsActiveMember | Active member (1/0) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target — churned (1) or stayed (0)** |

---

## ⚙️ Preprocessing Pipeline

- **Label Encoding** — Gender (Male/Female → 0/1)
- **One-Hot Encoding** — Geography (France/Germany/Spain → 3 binary columns)
- **Standard Scaling** — All numerical features normalized using `StandardScaler`

---

## 🗂️ Project Structure

```
ANN-Classification/
│
├── app.py                        # Streamlit web app
├── experiments.ipynb             # Model training notebook
├── prediction.ipynb              # Single prediction notebook
│
├── model.h5                      # Trained ANN model
├── scaler.pkl                    # Fitted StandardScaler
├── onehot_encoder_geo.pkl        # Fitted OneHotEncoder for Geography
├── label_encoder_gender.pkl      # Fitted LabelEncoder for Gender
│
├── requirements.txt              # Python dependencies
├── .python-version               # Python 3.11 pin for Streamlit Cloud
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Krish4055/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

**2. Create and activate environment**
```bash
conda create -n annenv python=3.11
conda activate annenv
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
tensorflow==2.21.0
scikit-learn
pandas
numpy
```

---

## 🔍 How It Works

1. User inputs customer details via the Streamlit UI
2. Geography is one-hot encoded using the saved `OneHotEncoder`
3. Gender is label encoded using the saved `LabelEncoder`
4. All features are scaled using the saved `StandardScaler`
5. The ANN model predicts churn probability (0 to 1)
6. Result displayed with a color-coded card and probability bar

---

## 📈 Sample Prediction

| Input | Value |
|---|---|
| Geography | Germany |
| Gender | Female |
| Age | 60 |
| Credit Score | 420 |
| Balance | $125,000 |
| Products | 1 |
| Active Member | No |

**Output: Will Churn — 82.4% probability** 🔴

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-yellow)

---

## 👨‍💻 Author

**Krish** — BE Information Technology, Year 3rd   
GitHub: [@Krish4055](https://github.com/Krish4055)
