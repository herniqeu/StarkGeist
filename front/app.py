import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
import os
import streamlit_shadcn_ui as ui
import altair as alt

# Suprimir avisos
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

# CSS code to hide Streamlit's menu and header
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

# Apply the CSS styles using markdown with HTML allowed
st.markdown(hide_st_style, unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('df_filtradoV2.1.csv')

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Funções de previsão ARIMA
def arima_forecast(data, column, steps=100):
    if not pd.api.types.is_numeric_dtype(data[column]):
        st.error(f"A coluna {column} não é numérica. Por favor, selecione uma coluna numérica para a previsão ARIMA.")
        return None
    
    try:
        model = ARIMA(data[column], order=(1, 1, 2))
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
        try:
            st.info("Carregando modelo existente...")
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {str(e)}")
            st.error("Stacktrace:")
            st.code(traceback.format_exc())
            return None
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
model_type = st.sidebar.radio("Selecione o modelo de previsão:", ('Série Temporal (ARIMA)', 'Rede Neural (What-If)', 'Dashboard'))
# choice = ui.select(options=['Série Temporal (ARIMA)', 'Rede Neural (What-If)', 'Dashboard'], label='Selecione o modelo de previsão:')

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

# if model_type == 'Série Temporal (ARIMA)':
#     st.header('Previsões de Série Temporal (ARIMA)')
    
#     # Select only numeric columns for forecasting
#     numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#     variable = st.selectbox('Escolha a variável para previsão:', numeric_columns)
    
#     # Get the ARIMA forecast
#     forecast = arima_forecast(df, variable)
    
#     if forecast is not None:
#         # Create a DataFrame for the forecast data
#         last_date = df['Date'].iloc[-1]  # Get the last date from the historical data
#         forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast))
#         forecast_df = pd.DataFrame({
#             'Date': forecast_dates,
#             'Value': forecast,
#             'Type': 'Previsão'
#         })
        
#         # Create a DataFrame for the historical data
#         historical_df = df[['Date', variable]].rename(columns={variable: 'Value'})
#         historical_df['Type'] = 'Histórico'
        
#         # Combine both DataFrames
#         combined_df = pd.concat([historical_df, forecast_df])
        
#         # Vega-Lite plot
#         chart = alt.Chart(combined_df).mark_line().encode(
#             x=alt.X('Date:T', title='Data'),
#             y=alt.Y('Value:Q', title='Valor'),
#             color='Type:N'
#         ).properties(
#             width=600,
#             height=400,
#             title=f'Previsão ARIMA para {variable}'
#         )
        
#         # Display the chart in Streamlit
#         st.altair_chart(chart, use_container_width=True)

elif model_type == 'Rede Neural (What-If)':
    st.header('Análise What-If com Rede Neural')
    
    X_scaled, y_scaled, scaler_X, scaler_y, input_columns, output_columns = prepare_nn_data(df)
    model = load_nn_model()
    
    if model is not None:
        st.sidebar.header('Ajuste as variáveis de entrada:')
        input_values = {}
        for col in input_columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # Add a small offset if min and max are the same
            if min_val == max_val:
                max_val += 0.000001
            
            default_value = float(df[col].iloc[-1])
            
            # Ensure default value is within the range
            default_value = max(min_val, min(max_val, default_value))
            
            input_values[col] = st.sidebar.slider(f'{col}:', 
                                                  min_val,
                                                  max_val, 
                                                  default_value)
        
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

elif model_type == 'Dashboard':
    value = ui.tabs(options=['Dashboard', 'Previsões', 'What-If'], default_value='Dashboard', key="kanaries")
    st.header(value)

    if value == "Dashboard":
        # ui.element("img", src="https://pub-8e7aa5bf51e049199c78b4bc744533f8.r2.dev/pygwalker-banner.png", className="w-full")
        # ui.element("link_button", text=value + " Github", url="https://github.com/Kanaries/pygwalker", className="mt-2", key="btn2")

        # Define mocked data for each column
        mock_data = {
            'Receita': {'value': "$50,000.00", 'description': "+10.5% from last month"},
            'Lucro Líquido': {'value': "$12,000.00", 'description': "+8.2% from last month"},
            'Despesas Operacionais': {'value': "$8,500.00", 'description': "-5.4% from last month"},
            'EBITDA': {'value': "$15,000.00", 'description': "+6.8% from last month"},
            'Endividamento': {'value': "$30,000.00", 'description': "+2.3% from last month"},
        }

        # Create columns dynamically based on the number of output columns
        cols = st.columns(len(mock_data))

        # Display metric cards for each output column with mocked data
        for idx, (key, data) in enumerate(mock_data.items()):
            with cols[idx]:
                ui.metric_card(
                    title=key,
                    content=data['value'],
                    description=data['description'],
                    key=f"card_{idx}"
                )

    elif value == "Previsões":
        # ui.element("img", src="https://pub-8e7aa5bf51e049199c78b4bc744533f8.r2.dev/graphic-walker-banner.png", className="w-full")
        # ui.element("link_button", text=value + " Github", url="https://github.com/Kanaries/graphic-walker", className="mt-2", key="btn2")

        # st.header('Previsões de Série Temporal (ARIMA)')
        
        # Select only numeric columns for forecasting
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        variable = st.selectbox('Escolha a variável para previsão:', numeric_columns)
        
        # Get the ARIMA forecast
        forecast = arima_forecast(df, variable)
        
        if forecast is not None:
            # Create a DataFrame for the forecast data
            last_date = df['Date'].iloc[-1]  # Get the last date from the historical data
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast))
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Value': forecast,
                'Type': 'Previsão'
            })
            
            # Create a DataFrame for the historical data
            historical_df = df[['Date', variable]].rename(columns={variable: 'Value'})
            historical_df['Type'] = 'Histórico'
            
            # Combine both DataFrames
            combined_df = pd.concat([historical_df, forecast_df])
            
            # Vega-Lite plot
            chart = alt.Chart(combined_df).mark_line().encode(
                x=alt.X('Date:T', title='Data'),
                y=alt.Y('Value:Q', title='Valor'),
                color='Type:N'
            ).properties(
                width=600,
                height=400,
                title=f'Previsão ARIMA para {variable}'
            )
            
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

    elif value == "What-If":
        ui.element("img", src="https://pub-8e7aa5bf51e049199c78b4bc744533f8.r2.dev/gwalkr-banner.png", className="w-full")
        ui.element("link_button", text=value + " Github", url="https://github.com/Kanaries/gwalkr", className="mt-2", key="btn2")
    # st.write("Selecionado:", value)

st.sidebar.markdown("---")
st.sidebar.info("Esta plataforma utiliza modelos de série temporal (ARIMA) e redes neurais para fazer previsões financeiras e análises 'what-if'.")