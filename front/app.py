import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Financial Predictor")

st.title("Financial Predictor")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df_filtrado = df.copy()  # Assuming df_filtrado is the same as df in this context

    # Drop 'Date' column if it exists
    if 'Date' in df_filtrado.columns:
        df_filtrado = df_filtrado.drop(['Date'], axis=1)

    # Define input and output columns
    input_columns = df_filtrado.columns.tolist()
    output_columns = ['Receita', 'Lucro Líquido', 'Despesas Operacionais', 'EBITDA', 'Endividamento']

    # Prepare input (X) and output (y) data
    X = df_filtrado[input_columns].values[:-1]
    y = df_filtrado[output_columns].values[1:]

    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(5)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Training section
    st.header("Model Training")
    epochs = st.slider("Number of epochs", min_value=10, max_value=1000, value=100, step=10)
    train_button = st.button("Train Model")

    if train_button:
        with st.spinner("Training in progress..."):
            history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
        
        st.success("Training completed!")

        # Plot training history
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

    # Prediction section
    st.header("Make Predictions")
    prediction_input = {}
    for col in input_columns:
        prediction_input[col] = st.number_input(f"Enter value for {col}", value=0.0)

    predict_button = st.button("Make Prediction")

    if predict_button:
        input_data = np.array([list(prediction_input.values())])
        input_scaled = scaler_X.transform(input_data)
        prediction_scaled = model.predict(input_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        st.subheader("Prediction Results:")
        for i, col in enumerate(output_columns):
            st.write(f"{col}: {prediction[0][i]:.2f}")

else:
    st.info("Please upload a CSV file to get started.")