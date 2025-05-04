# Step 1: Ignore Warnings and Import Libraries
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Load IMDB Dataset
# Only keep the top 10,000 most frequent words
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# Step 3: Preprocess Data - Pad Sequences
max_length = 200
x_train_padded = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test_padded = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

print(f"Padded x_train shape: {x_train_padded.shape}")

# Step 4: Build the Deep Neural Network Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Train the Model
history = model.fit(
    x_train_padded, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Step 6: Evaluate the Model
loss, accuracy = model.evaluate(x_test_padded, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Step 7: Plot Training & Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Predict on Sample Review
# Let's decode the review to see what it says
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

sample_review = x_test[0]
print("Review Text:", decode_review(sample_review))
print("Actual Sentiment:", "Positive" if y_test[0] == 1 else "Negative")

# Predict sentiment
sample_review_padded = pad_sequences([sample_review], maxlen=max_length, padding='post', truncating='post')
prediction = model.predict(sample_review_padded)[0][0]
print(f"Predicted Sentiment Score: {prediction:.4f}")
print("Predicted Sentiment:", "Positive" if prediction >= 0.5 else "Negative")

# Step 9: Display Accuracy as Percentage
print(f"\nModel Test Accuracy: {accuracy * 100:.2f}%")

# Step 10: Optional: Detailed Accuracy Report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predict on test set
y_pred_probs = model.predict(x_test_padded)
y_pred = (y_pred_probs >= 0.5).astype(int).reshape(-1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
