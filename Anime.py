# 🎯 Anime Genre Prediction using Simple RNN + NLTK Preprocessing
# This Neural Network model predicts anime genres based on their synopsis.

# 📦 Imports
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 🧠 NLTK Setup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

# 🧹 Text Cleaning Function
def clean_text_nltk(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text, preserve_line=True)
    tokens = [t for t in tokens if t not in STOP]
    tokens = [LEM.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# 📄 Load Dataset
df = pd.read_csv(r"C:\Users\akjee\Documents\AI\NLP\NLP - DL\RNN\anime_recommendation_dataset.csv")
df = df[['synopsis', 'genres']].dropna().reset_index(drop=True)

# 🧼 Clean Synopsis
df['synopsis_clean'] = df['synopsis'].apply(clean_text_nltk)

# 🎯 Convert Genre Strings to Lists
df['genre_list'] = df['genres'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])

# 🔍 Filter Rare Genres
min_count = 10
all_genres = pd.Series([g for sub in df['genre_list'] for g in sub])
common_genres = set(all_genres.value_counts()[all_genres.value_counts() >= min_count].index)
df['genre_list'] = df['genre_list'].apply(lambda gl: [g for g in gl if g in common_genres])
df = df[df['genre_list'].map(len) > 0].reset_index(drop=True)

# 🔢 Multi-label Binarization
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['genre_list'])
genre_names = mlb.classes_
num_classes = len(genre_names)
print("✅ Found classes:", num_classes, genre_names)

# 🔠 Tokenization
NUM_WORDS = 10000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['synopsis_clean'])
X_seq = tokenizer.texts_to_sequences(df['synopsis_clean'])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# 🧪 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# 🧠 Build Simple RNN Model
EMBED_DIM = 64
RNN_UNITS = 64
DROPOUT = 0.3

model = Sequential([
    Embedding(NUM_WORDS, EMBED_DIM, input_length=MAX_LEN),
    Dropout(DROPOUT),
    SimpleRNN(RNN_UNITS),
    Dropout(DROPOUT),
    Dense(64, activation='relu'),
    Dropout(DROPOUT),
    Dense(num_classes, activation='sigmoid')  # Multi-label output
])

model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 🛑 Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# 🏋️ Train Model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=25, batch_size=128, callbacks=[early_stop])

# 📈 Plot Training Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 📊 Evaluate Model
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

print("📌 Sample-based F1 score:", f1_score(y_test, y_pred, average='samples'))
print("📌 Classification Report:")
print(classification_report(y_test, y_pred, target_names=genre_names, zero_division=0))

# 🔮 Genre Prediction Function
def predict_genres(text, thresh=0.5, top_k=3):
    t = clean_text_nltk(text)
    seq = tokenizer.texts_to_sequences([t])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    probs = model.predict(pad)[0]
    idxs = np.where(probs >= thresh)[0]
    if len(idxs) == 0:
        idxs = probs.argsort()[-top_k:][::-1]
    return [(genre_names[i], float(probs[i])) for i in idxs]

# 🧪 Try Example
example = "A young hero trains to save his village while discovering ancient powers."
print("🔮 Predicted genres:", predict_genres(example, thresh=0.4))
