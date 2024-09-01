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
import locale

# Suprimir avisos
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

# Combined CSS to hide Streamlit's menu and header, and customize sliders
custom_styles = """
    <style>
    /* Hide Streamlit menu and header */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

# Apply the combined CSS styles using markdown with HTML allowed
st.markdown(custom_styles, unsafe_allow_html=True)


try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    st.error("Locale setting 'pt_BR.UTF-8' is not supported on this system. Please ensure your environment supports this locale.")

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
            # st.info("Carregando modelo existente...")
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
st.title('GEIST: Plataforma de Previsão Financeira')

value = ui.tabs(options=['Dashboard', 'Previsões', 'What-If'], default_value='Dashboard', key="kanaries")
st.header(value)

if value == "Dashboard":

    try:
        # Prepare data and load the model as in the What-If section
        X_scaled, y_scaled, scaler_X, scaler_y, input_columns, output_columns = prepare_nn_data(df)
        model = load_nn_model()

        if model is not None:
            # Use the latest row from the dataset as the input
            input_data = df[input_columns].iloc[-1].values
            input_data_scaled = scaler_X.transform(input_data.reshape(1, -1))
            
            # Get predictions using the loaded model
            prediction = nn_predict(model, input_data_scaled, scaler_y)

            # Create columns dynamically based on the number of output columns
            cols = st.columns(len(output_columns))

            # Display each prediction in a metric card as formatted Brazilian Real
            for idx, col in enumerate(output_columns):
                with cols[idx]:
                    # Format the content as Brazilian Real currency
                    content = locale.currency(prediction[idx], symbol=True, grouping=True)

                    # Ensure R$ is positioned correctly
                    if content.endswith("R$"):
                        content = "R$ " + content.replace(" R$", "")

                    ui.metric_card(
                        title=col,
                        content=content,  # Display as currency with R$ in front
                        description=f"Previsão para {col}",
                        key=f"card_{idx}"
                    )

        else:
            raise Exception("Não foi possível carregar o modelo.")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou processar dados: {str(e)}")

elif value == "Previsões":
    st.write('Previsões de Série Temporal (ARIMA)')

    # Select only numeric columns for forecasting, excluding 'Close_GSPC'
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 'Close_GSPC' from the list if it exists
    if 'Close_GSPC' in numeric_columns:
        numeric_columns.remove('Close_GSPC')

    # Create a dropdown for selecting the variable
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
    st.write('Análise What-If com Rede Neural')

    # Inicialize um contador de tentativas na sessão do Streamlit
    if 'retry_count' not in st.session_state:
        st.session_state.retry_count = 0

    try:
        X_scaled, y_scaled, scaler_X, scaler_y, input_columns, output_columns = prepare_nn_data(df)
        model = load_nn_model()
        
        if model is not None:
            # Resetar o contador de tentativas se o modelo for carregado com sucesso
            st.session_state.retry_count = 0

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

            # Create columns dynamically based on the number of output columns
            cols = st.columns(len(output_columns))

            # Display each prediction in a metric card as formatted Brazilian Real
            for idx, col in enumerate(output_columns):
                with cols[idx]:
                    # Format the content as Brazilian Real currency
                    content = locale.currency(prediction[idx], symbol=True, grouping=True)

                    # Ensure R$ is positioned correctly
                    if content.endswith("R$"):
                        content = "R$ " + content.replace(" R$", "")
                    
                    ui.metric_card(
                        title=col,
                        content=content,  # Display as currency with R$ in front
                        description=f"Previsão para {col}",
                        key=f"card_{idx}"
                    )
            
            # Prepare the data for Altair
            data = pd.DataFrame({
                'Output': output_columns,
                'Value': prediction
            })

            # Create a Vega-Lite bar chart using Altair
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('Output', title='Output Columns', sort=None),
                y=alt.Y('Value', title='Valor'),
                tooltip=['Output', 'Value']
            ).properties(
                title='Previsões da Rede Neural',
                width=600,
                height=400
            ).configure_axisX(
                labelAngle=45  # Rotate x-axis labels
            )

            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)
        else:
            raise Exception("Não foi possível carregar o modelo.")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou processar dados: {str(e)}")
        
        # Incrementar o contador de tentativas
        st.session_state.retry_count += 1
        
        # Limitar o número de tentativas para evitar loops infinitos
        if st.session_state.retry_count < 3:
            st.warning("Tentando recarregar a página em 5 segundos...")
            time.sleep(5)
            st.experimental_rerun()
        else:
            st.error("Falha ao carregar o modelo após várias tentativas. Por favor, verifique o arquivo do modelo e tente novamente mais tarde.")
            st.session_state.retry_count = 0 

# st.sidebar.markdown("---")
st.sidebar.info("Esta plataforma utiliza modelos de série temporal (ARIMA) e redes neurais para fazer previsões financeiras e análises 'what-if'.")