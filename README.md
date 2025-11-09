# 🎌 Anime Genre Prediction using Simple RNN + NLTK

Predict anime genres automatically from synopsis text using a deep learning model built with TensorFlow/Keras.  
This project cleans plot synopses with NLTK, tokenizes data, and trains a multi-label classifier using a Simple RNN network.

---

## 🚀 Features
✅ NLTK-based text preprocessing  
✅ Multi-label genre classification  
✅ Simple RNN model with embedding layer  
✅ Genre threshold + Top-K inference  
✅ Automatic filtering of rare genres  
✅ Evaluation using F1-score & classification report  

---

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- NLTK
- NumPy / Pandas
- Scikit-Learn
- Matplotlib

---

## 🧠 Model Architecture
- Tokenizer + padded sequences  
- Embedding (64-dim)
- Simple RNN (64 units)
- Dense (Relu)
- Dense (sigmoid output layer)

Output is multi-label — each anime can belong to multiple genres.

---

## 📦 Installation

### Clone repo
git clone https://github.com/<your-username>/anime-genre-prediction-rnn.git
cd anime-genre-prediction-rnn

### Install dependencies
pip install -r requirements.txt

## 🧼 Dataset

The dataset must include:

synopsis, genres features or columns

## 🔧 Training

The script performs:

Text cleaning (stopwords, lemmatization)

Tokenization

Train/test split

RNN training with early stopping

Evaluation in F1 + report

## 📊 Evaluation

Sample-based F1 score

Classification report per genre

## ✅ Improvements / Next Steps

Replace SimpleRNN with LSTM/GRU

Use pre-trained embeddings (GloVe / FastText)

Try transformer-based models (BERT)

Hyperparameter tuning

Web API frontend

## 🤝 Contributing

Pull requests and feature requests are welcome!

## 📄 License

MIT License

---
