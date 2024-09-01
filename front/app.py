import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
import os

# Suprimir avisos
warnings.filterwarnings("ignore")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('df_filtrado.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Funções de previsão ARIMA
def arima_forecast(data, column, steps=100):
    if not pd.api.types.is_numeric_dtype(data[column]):
        st.error(f"A coluna {column} não é numérica. Por favor, selecione uma coluna numérica para a previsão ARIMA.")
        return None
    
    try:
        model = ARIMA(data[column], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        st.error(f"Erro ao fazer a previsão ARIMA: {str(e)}")
        return None

# Funções de previsão da Rede Neural
def prepare_nn_data(df):
    input_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    output_columns = ['Receita', 'Lucro Líquido', 'Despesas Operacionais', 'EBITDA', 'Endividamento']
    
    X = df[input_columns].values[:-1]
    y = df[output_columns].values[1:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, input_columns, output_columns

def load_nn_model():
    model_path = "stark_geist_model.keras"
    if os.path.exists(model_path):
        st.info("Carregando modelo existente...")
        model = load_model(model_path)
        return model
    else:
        st.error("Modelo não encontrado. Por favor, certifique-se de que o arquivo 'stark_geist_model.keras' está no diretório correto.")
        return None

def nn_predict(model, input_data, scaler_y):
    prediction_scaled = model.predict(input_data.reshape(1, -1))
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return prediction[0]

# Interface Streamlit
st.title('Plataforma de Previsão Financeira')

# Sidebar para seleção de modelo
model_type = st.sidebar.radio("Selecione o modelo de previsão:", ('Série Temporal (ARIMA)', 'Rede Neural (What-If)'))

if model_type == 'Série Temporal (ARIMA)':
    st.header('Previsões de Série Temporal (ARIMA)')
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    variable = st.selectbox('Escolha a variável para previsão:', numeric_columns)
    
    forecast = arima_forecast(df, variable)
    
    if forecast is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df[variable], label='Histórico')
        ax.plot(pd.date_range(start=df['Date'].iloc[-1], periods=len(forecast)), forecast, label='Previsão')
        ax.set_title(f'Previsão ARIMA para {variable}')
        ax.legend()
        st.pyplot(fig)

else:
    st.header('Análise What-If com Rede Neural')
    
    X_scaled, y_scaled, scaler_X, scaler_y, input_columns, output_columns = prepare_nn_data(df)
    model = load_nn_model()
    
    if model is not None:
        st.sidebar.header('Ajuste as variáveis de entrada:')
        input_values = {}
        for col in input_columns:
            default_value = df[col].iloc[-1]
            input_values[col] = st.sidebar.slider(f'{col}:', 
                                                  float(df[col].min()), 
                                                  float(df[col].max()), 
                                                  float(default_value))
        
        input_data = np.array([input_values[col] for col in input_columns])
        input_data_scaled = scaler_X.transform(input_data.reshape(1, -1))
        
        prediction = nn_predict(model, input_data_scaled, scaler_y)
        
        st.subheader('Previsões baseadas nos inputs:')
        for i, col in enumerate(output_columns):
            st.metric(label=col, value=f"{prediction[i]:.2f}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(output_columns, prediction)
        ax.set_title('Previsões da Rede Neural')
        ax.set_ylabel('Valor')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Não foi possível carregar o modelo. Por favor, verifique se o arquivo do modelo está presente.")

st.sidebar.markdown("---")
st.sidebar.info("Esta plataforma utiliza modelos de série temporal (ARIMA) e redes neurais para fazer previsões financeiras e análises 'what-if'.")