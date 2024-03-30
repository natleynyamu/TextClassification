import streamlit as st
import numpy as np
import tensorflow as tf
import pickle


# Load the model
model = tf.keras.models.load_model("model/lstm_model.h5")


# Load the tokenizer from the file
with open('model/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# The max_length from the training process
max_length = 62

st.title("🗨️ Sentiment and Text Classification for Product Rating Optimization 🗨️")

def predict_text_rating(text_input):
    # Tokenize and pad the text input to prepare for prediction
    sequences = tokenizer.texts_to_sequences([text_input])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)

    # Predict using the trained model
    predictions = model.predict(padded_sequences)

    # Since the output layer uses softmax, `predictions` will give the probabilities for each class
    predicted_class = np.argmax(predictions, axis=1)

    # Translate the predicted class index into the corresponding rating
    predicted_rating = predicted_class + 1  # Adjusting because our classes were zero-indexed
    return predicted_rating

# Streamlit app interface
st.write("### Enter your review text below to predict the product rating:")
user_input = st.text_area("Review Text", placeholder="Type here...")

if st.button("Predict Rating"):
    if not user_input:  # Check if user_input is empty
        st.error("Please enter a review before predicting the rating.")
    else:
        output = predict_text_rating(user_input)
        if output[0] <= 2:
            st.write(f"Predicted Rating: {output[0]} 👎")  # Output[0] to unpack the numpy array and thumbs-down emoji
        else:
            st.write(f"Predicted Rating: {output[0]} 👍")  # Output[0] to unpack the numpy array and thumbs-up emoji
