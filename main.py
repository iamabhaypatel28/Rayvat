import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import random


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.lower()

def preprocess_text(text, max_vocab_size=5000, seq_length=10):
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    total_words = min(len(tokenizer.word_index) + 1, max_vocab_size)
    
    input_sequences = []
    words = text.split()
    for i in range(len(words) - seq_length):
        sequence = words[i:i+seq_length]
        input_sequences.append(sequence)
    
    sequences = tokenizer.texts_to_sequences([" ".join(seq) for seq in input_sequences])
    sequences = pad_sequences(sequences, maxlen=seq_length, padding='pre')
    
    X, y = sequences[:, :-1], sequences[:, -1]
    y = np.array(y)  
    return X, y, tokenizer, seq_length, total_words

# model
def build_model(total_words, seq_length):
    model = Sequential([
        Embedding(total_words, 100, input_length=seq_length-1),
        LSTM(150, return_sequences=True),
        LSTM(100),
        Dense(total_words, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Generate text
def generate_text(model, tokenizer, seed_text, max_words, seq_length):
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]
        output_word = tokenizer.index_word.get(predicted, "")
        seed_text += " " + output_word
    return seed_text

if __name__ == "__main__":
    data_path = r"D:\rayvat\Rayvat\data\shakespeare.txt"
    text_data = load_data(data_path)
    X, y, tokenizer, seq_length, total_words = preprocess_text(text_data)
    
    model = build_model(total_words, seq_length)
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=20, verbose=1, callbacks=[early_stop])
    
    seed_text = "to be or not to be"
    generated_text = generate_text(model, tokenizer, seed_text, 50, seq_length)
    print("Generated Text:", generated_text)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_text_gen_model.h5")
