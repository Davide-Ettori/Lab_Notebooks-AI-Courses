import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import pickle
import matplotlib.pyplot as plt

word2vec = pickle.load(open('../models/twitter.pkl', 'rb'))

def get_embedding(word):   
    return word2vec[word] if word in word2vec else np.zeros(word2vec.vector_size)

def text_to_average_embedding(texts):
    embeddings = []
    for text in texts:
        avg_embedding = np.mean([get_embedding(word) for word in text], axis=0)
        embeddings.append(avg_embedding)
    return np.array(embeddings)

data = pd.read_csv('../data_cleaned/train_data_cleaned.csv')

initial_count = len(data)
data = data.dropna()
final_count = len(data)
print(f'Number of rows with null values dropped: {initial_count - final_count}')

tweets = data['tweet'] 
labels = data['sentiment']  

X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=42)

y_train += 1
y_test += 1

X_train = pd.Series(X_train).fillna('').astype(str)
X_test = pd.Series(X_test).fillna('').astype(str)

X_train_embeddings = text_to_average_embedding(X_train)
X_test_embeddings = text_to_average_embedding(X_test)

model = models.Sequential([
    layers.InputLayer(input_shape=(word2vec.vector_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_embeddings, y_train, validation_split=0.2, epochs=25, batch_size=32, verbose=2) 

test_loss, test_accuracy = model.evaluate(X_test_embeddings, y_test)
print(f'Final Test Accuracy: {test_accuracy:.4f}')

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()