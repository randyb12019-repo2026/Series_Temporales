import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import io
import sys

st.set_page_config(page_title="Series Temporales - NVDA", layout="wide")

st.title("Series Temporales - Análisis de NVIDIA (NVDA)")
st.markdown("Proyecto educativo de análisis de series temporales con Python.")

@st.cache_data
def load_data():
    df = yf.download("NVDA", start="2020-01-01", end="2026-06-04")
    df.columns = df.columns.get_level_values(0)
    df["Precio_Medio"] = (df["High"] + df["Low"]) / 2
    return df

with st.spinner("Descargando datos de NVDA..."):
    nvidia = load_data()

st.success("Datos cargados correctamente")

st.sidebar.header("Opciones")
seccion = st.sidebar.radio(
    "Ir a:",
    [
        "Vista general",
        "Descomposición estacional",
        "Medias móviles",
        "Test de estacionariedad",
        "Modelo ARIMA",
        "Pronóstico y evaluación",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Realizado por:**  \n"
    "Randy Bonucci  \n"
    "Bootcamp Data Analysis & IA  \n"
    "**Upgrade Hub**  \n"
    "[GitHub](https://github.com/randyb12019-repo2026/Series_Temporales)"
)

if seccion == "Vista general":
    st.header("Vista general de los datos")
    st.dataframe(nvidia.head(10))
    st.subheader("Estadísticas descriptivas")
    st.dataframe(nvidia.describe())

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(nvidia.index, nvidia["Close"], label="Close", color="blue", alpha=0.7)
    ax.plot(nvidia.index, nvidia["Precio_Medio"], label="Precio Medio", color="purple", alpha=0.7)
    ax.set_title("Precio de Cierre y Precio Medio de NVDA")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.info(
        "**Conclusión:** NVDA ha mostrado un crecimiento exponencial desde 2020, "
        "pasando de ~6 USD a superar los 130 USD en 2026. Se observa una alta volatilidad "
        "en los datos diarios, con una tendencia alcista muy marcada que sugiere que la serie "
        "no es estacionaria en su forma original."
    )

elif seccion == "Descomposición estacional":
    st.header("Descomposición estacional")
    serie = nvidia["Close"].resample("ME").mean()

    fig = seasonal_decompose(serie, model="additive", period=12).plot()
    fig.set_size_inches(12, 8)
    st.pyplot(fig)

    st.info(
        "**Conclusión:** La tendencia confirma el crecimiento sostenido de NVDA. "
        "La componente estacional muestra patrones anuales repetitivos, aunque de menor magnitud "
        "relativa frente a la tendencia. El residuo presenta mayor variabilidad en periodos de alta "
        "volatilidad del mercado, lo cual es esperable en activos tecnológicos."
    )

elif seccion == "Medias móviles":
    st.header("Medias móviles")

    ventanas = st.multiselect(
        "Selecciona ventanas para las medias móviles:",
        [7, 30, 60, 90],
        default=[7, 30, 90],
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        nvidia.index,
        nvidia["Open"],
        label="Precio de Apertura",
        color="blue",
        alpha=0.4,
    )
    for w in ventanas:
        mm = nvidia["Open"].rolling(window=w).mean()
        ax.plot(nvidia.index, mm, label=f"Media Móvil {w} días")
    ax.set_title("Precio de Apertura de NVDA con Medias Móviles")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.info(
        "**Conclusión:** Las medias móviles de ventana más larga (90 días) suavizan mejor el ruido "
        "diario y revelan la tendencia subyacente. Las de ventana corta (7 días) siguen de cerca "
        "la volatilidad diaria. La diferenciación elimina la tendencia y estabiliza la media, "
        "lo que es un paso clave para modelar con ARIMA."
    )

    st.header("Precio Medio con diferenciación")
    nvidia_mensual = nvidia.resample("ME").mean()
    nvidia_mensual["MM30"] = nvidia_mensual["Precio_Medio"].rolling(window=30, min_periods=1).mean()
    diferenciacion = nvidia_mensual["MM30"].diff()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(
        nvidia_mensual.index,
        nvidia_mensual["Precio_Medio"],
        label="Precio Medio Mensual",
        color="purple",
    )
    ax2.plot(
        nvidia_mensual.index,
        diferenciacion,
        label="Diferenciación",
        color="red",
    )
    ax2.set_title("Precio Medio Mensual de NVDA y su Diferenciación")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Precio (USD) / Cambio")
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

elif seccion == "Test de estacionariedad":
    st.header("Test de Dickey-Fuller Aumentado (ADF)")

    serie = nvidia["Precio_Medio"].dropna()
    serie_dif = serie.diff().dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Serie original")
        res = adfuller(serie)
        st.write(f"**Estadístico ADF:** {res[0]:.4f}")
        st.write(f"**p-valor:** {res[1]:.4f}")
        st.write(f"**Nº observaciones:** {res[3]}")
        if res[1] <= 0.05:
            st.success("Veredicto: ES estacionaria")
        else:
            st.error("Veredicto: NO es estacionaria (conviene diferenciar)")

    with col2:
        st.subheader("Serie diferenciada")
        res_dif = adfuller(serie_dif)
        st.write(f"**Estadístico ADF:** {res_dif[0]:.4f}")
        st.write(f"**p-valor:** {res_dif[1]:.4f}")
        st.write(f"**Nº observaciones:** {res_dif[3]}")
        if res_dif[1] <= 0.05:
            st.success("Veredicto: ES estacionaria")
        else:
            st.error("Veredicto: NO es estacionaria (conviene diferenciar)")

    st.info(
        "**Conclusión:** El p-valor alto (>0.05) en la serie original confirma que no es estacionaria, "
        "lo cual es esperable por la tendencia alcista. Al aplicar la primera diferencia, el p-valor "
        "cae a ~0.00, indicando que la serie diferenciada es estacionaria. Esto valida el uso de "
        "d=1 en el modelo ARIMA."
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(serie.index, serie, label="Original", alpha=0.7)
    ax.plot(serie_dif.index, serie_dif, label="Diferenciada", alpha=0.7)
    ax.set_title("Precio Medio de NVDA - Original vs Diferenciada")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

elif seccion == "Modelo ARIMA":
    st.header("Modelo ARIMA Automático (pmdarima)")

    serie = nvidia["Precio_Medio"].dropna()
    n_test = max(1, int(len(serie) * 0.2))
    train = serie.iloc[:-n_test]
    test = serie.iloc[-n_test:]

    with st.spinner("Entrenando modelo ARIMA..."):
        modelo = pm.auto_arima(
            train, seasonal=False, stepwise=True, trace=False
        )

    st.subheader("Resumen del modelo")
    buf = io.StringIO()
    modelo.summary().as_csv()
    st.text(str(modelo.summary()))

    predicciones = modelo.predict(n_periods=n_test)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train, label="Train", color="blue")
    ax.plot(test.index, test, label="Test", color="orange")
    ax.plot(test.index, predicciones, label="Predicciones", color="green")
    ax.set_title("Predicciones Auto ARIMA vs Test")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    mae = mean_absolute_error(test, predicciones)
    rmse = np.sqrt(mean_squared_error(test, predicciones))
    st.metric("MAE", f"{mae:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")

    st.info(
        "**Conclusión:** El modelo ARIMA no estacional captura razonablemente la tendencia, "
        "pero subestima los picos de volatilidad. Las métricas MAE y RMSE reflejan el error "
        "promedio de las predicciones. Para series financieras con alta volatilidad, "
        "un modelo SARIMA estacional suele ofrecer mejores resultados."
    )

elif seccion == "Pronóstico y evaluación":
    st.header("Pronóstico y evaluación con SARIMA estacional")

    serie = nvidia["Precio_Medio"].dropna()
    n_test = max(1, int(len(serie) * 0.2))
    train = serie.iloc[:-n_test]
    test = serie.iloc[-n_test:]

    with st.spinner("Entrenando modelo SARIMA estacional..."):
        modelo = pm.auto_arima(
            train, seasonal=True, m=12, stepwise=True, trace=False
        )

    st.subheader("Resumen del modelo SARIMA")
    st.text(str(modelo.summary()))

    predicciones = modelo.predict(n_periods=n_test)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train, label="Train", color="blue")
    ax.plot(test.index, test, label="Test", color="orange")
    ax.plot(test.index, predicciones, label="Predicciones", color="green")
    ax.set_title("Predicciones SARIMA vs Test")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    mae = mean_absolute_error(test, predicciones)
    rmse = np.sqrt(mean_squared_error(test, predicciones))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", f"{mae:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")

    st.info(
        "**Conclusión:** El modelo SARIMA estacional aprovecha los patrones periódicos "
        "anuales (m=12) para mejorar el pronóstico. Si las métricas MAE/RMSE son menores "
        "que en el modelo ARIMA no estacional, confirma que incluir la estacionalidad "
        "aporta valor predictivo. NVDA, al ser un activo tecnológico, muestra patrones "
        "estacionales asociados a ciclos de innovación y resultados trimestrales."
    )
