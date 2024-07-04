import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Definimos los parámetros de configuración de la aplicación
st.set_page_config(
    page_title="Predicción de series de tiempo con Prophet y LSTM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header('Predicción de series de tiempo con Prophet y LSTM')
st.warning('Se debe cargar un archivo csv cuya primera columna sea una fecha y la segunda sea un valor a predecir')
# Declaramos el control para cargar archivos
archivo_cargado = st.file_uploader("Elige un archivo", type='csv')

# Función para crear una secuencia de datos
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def calcular_metricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return rmse, mae, mape, r2, correlation

# Si existe un archivo cargado ejecutamos el código
if archivo_cargado is not None:
    # Cargamos el archivo CSV con pandas
    df = pd.read_csv(archivo_cargado)

    # Aseguramos que el DataFrame tiene al menos dos columnas
    if df.shape[1] < 2:
        st.error("El archivo CSV debe tener al menos dos columnas.")
    else:
        # Seleccionamos solo las primeras dos columnas
        df = df.iloc[:, :2]
        # Renombramos las columnas a como lo existe Prophet ds = Fecha, y= Valor
        df.columns = ['ds', 'y']
        
        # Definimos las frecuencias del control
        frequencias = ['Día', 'Semana', 'Mes', 'Año']
        # Definimos los códigos de cada frecuencia
        frequenciasCodigo = ['D', 'W', 'M', 'Y']
        # Definimos las columnas
        c1, c2 = st.columns([30, 70])
        with c1:
            # Mostramos el dataframe
            st.dataframe(df, use_container_width=True)
        with c2:
            # Mostramos el control de selección de frecuencias
            parFrecuencia = st.selectbox('Frecuencia de los datos', options=['Día', 'Semana', 'Mes', 'Año'])
            # Mostramos el control para seleccionar el horizonte de predicción
            parPeriodosFuturos = st.slider('Periodos a predecir', 5, 300, 5)
            # Botón para ejecutar la predicción
            btnEjecutarForecast = st.button('Ejecutar predicción')
        
        # Cuando se presione el botón ejecutamos el código
        if btnEjecutarForecast:
            # Cargamos el Prophet
            m = Prophet()
            # Ejecutamos el modelo
            m.fit(df)
            # Detectamos la frecuencia entregada
            frequencia = frequenciasCodigo[frequencias.index(parFrecuencia)]
            # Generamos la predicción de acuerdo a la frecuencia y los periodos solicitados
            future = m.make_future_dataframe(periods=parPeriodosFuturos, freq=frequencia)
            # Guardamos la predicción
            forecast = m.predict(future)
            # Sacamos a parte solo los valores de la predicción
            dfPrediccion = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(parPeriodosFuturos)
            # Generamos la gráfica de modelo Prophet
            fig1 = m.plot(forecast)
            # Generamos tabs o pestañas para mostrar gráficos y datos
            tab1, tab2, tab3, tab4 = st.tabs(['Resultado', 'Gráfico Prophet', 'Predicción LSTM Ajustada', 'Métricas de Desempeño'])
            # Asignamos al dataset df una columna Tipo que indique los datos reales
            df['Tipo'] = 'Real'
            # Asignamos al dataset dfPrediccion una columna Tipo que indique los datos de Predicción
            dfPrediccion['Tipo'] = 'Predicción'
            # Renombramos la columna yhat que retorna el modelo como y
            dfPrediccion = dfPrediccion.rename(columns={'yhat': 'y'})
            # Concatenamos los datos reales y la predicción
            dfResultado = pd.concat([df.sort_values(by='ds'), dfPrediccion[['ds', 'y', 'Tipo']]])
            with tab1:
                # En el primer tab mostramos la predicción completa
                c1, c2 = st.columns([30, 70])
                with c1:
                    st.dataframe(dfResultado)
                    # Convertimos el dataframe a CSV y lo guardamos en una variable
                    ArchivoCSV = dfResultado.to_csv(index=False).encode('utf-8')
                    # Creamos el nombre del nuevo archivo
                    archivoNuevo = archivo_cargado.name
                    archivoNuevo = f'prediccion_{archivoNuevo}'
                    # Usamos el botón de descarga de Streamlit
                    st.download_button(
                        label="Descargar resultado como CSV",  # Etiqueta del botón
                        data=ArchivoCSV,  # Datos a descargar
                        file_name=archivoNuevo,  # Nombre del archivo
                        mime='text/csv'  # Formato a descargar
                    )
                with c2:
                    # Mostramos el gráfico de los resultados de la predicción
                    fig = px.line(dfResultado, x='ds', y='y', color='Tipo')
                    st.plotly_chart(fig, use_container_width=True)
            with tab2:
                # En el tab2, mostramos la gráfica que genera Prophet
                st.write(fig1)
            
            with tab3:
                # Preparación de datos para LSTM
                df['ds'] = pd.to_datetime(df['ds'])
                df.set_index('ds', inplace=True)
                data = df['y'].values
                data = data.reshape(-1, 1)

                # Normalización de los datos
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data)

                # Creación de secuencias
                SEQ_LENGTH = 30
                X, y = create_sequences(data_scaled, SEQ_LENGTH)

                # División de los datos en entrenamiento y prueba
                TRAIN_SIZE = int(len(X) * 0.8)
                X_train, X_test = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
                y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]

                # Construcción del modelo LSTM
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
                    LSTM(50),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=10, batch_size=32)

                # Predicción con LSTM
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_test = scaler.inverse_transform(y_test)

                # Asegurarse de que las predicciones LSTM tengan el tamaño correcto
                y_pred_resized = np.resize(y_pred, forecast['yhat'].shape)

                # Ajuste de las predicciones de Prophet con LSTM
                forecast['yhat_adjusted'] = forecast['yhat'] + y_pred_resized.flatten()

                # Preparar datos para mostrar en gráfica
                df_pred = pd.DataFrame({
                    'ds': forecast['ds'],
                    'yhat': forecast['yhat'],
                    'yhat_adjusted': forecast['yhat_adjusted']
                })

                # Gráfico de resultados de LSTM ajustado
                fig_lstm = px.line(df_pred, x='ds', y=['yhat', 'yhat_adjusted'], labels={'value': 'Predicción', 'ds': 'Fecha'})
                fig_lstm.update_layout(title='Comparación de Predicciones Prophet y LSTM Ajustado')
                st.plotly_chart(fig_lstm, use_container_width=True)

            with tab4:
                # Calcular métricas de desempeño
                rmse, mae, mape, r2, correlation = calcular_metricas(y_test.flatten(), y_pred.flatten())

                # Mostrar métricas de desempeño
                metrics_df = pd.DataFrame({
                    'Métrica': ['Raíz Error Cuadrático Medio', 'Error Absoluto Medio', 'Error Relativo (%)', 'R cuadrado', 'Correlación'],
                    'Valor': [rmse, mae, mape, r2, correlation]
                })

                st.table(metrics_df)