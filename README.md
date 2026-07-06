# Series Temporales - Proyecto de Analisis Temporal

Actividad del Bootcamp de Data Analysis & IA en **Upgrade Hub**.

Proyecto educativo de analisis de series temporales con Python, usando datos financieros de NVIDIA (NVDA) via yfinance.

## Datos

- **Activo:** NVDA (NVIDIA Corporation)
- **Periodo:** 2020-01-01 a 2026-01-01
- **Frecuencia:** diaria, remuestreada a mensual

## Tecnicas aplicadas

1. Descomposicion estacional (seasonal_decompose)
2. Medias moviles (ventanas 7, 30, 60, 90 dias)
3. Remuestreo temporal (resample ME)
4. Modelo ARIMA automatico (pmdarima auto_arima)
5. Pronostico y evaluacion con metricas (MAE, RMSE)

## Estructura del proyecto

```
Series_Temporales/
├── app.py                            # Streamlit app para visualizacion interactiva
├── notebooks/
│   └── st.ipynb                      # Notebook principal con analisis temporal
├── Dockerfile                        # Imagen Python 3.13-slim con Jupyter
├── docker-compose.yml                # Orquestacion del contenedor
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Docker

### Construir y ejecutar

```bash
docker compose build
docker compose up
```

Esto inicia un contenedor con el notebook listo para usar.

### Ejecutar notebook dentro del contenedor

```bash
docker compose run --rm series_temporales jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

## Streamlit App

Se incluye una aplicacion interactiva desarrollada con **Streamlit** para visualizar los analisis de series temporales.

### Ejecutar localmente

```bash
streamlit run app.py
```

### Ejecutar con Docker

```bash
docker build -t series_temporales-streamlit .
docker run -p 8501:8501 series_temporales-streamlit streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Funcionalidades

- Vista general de los datos de NVDA
- Descomposicion estacional (tendencia, estacionalidad, residuo)
- Medias moviles configurables (7, 30, 60, 90 dias)
- Test de estacionariedad ADF
- Modelo ARIMA automatico con pmdarima
- Pronostico y evaluacion con metricas MAE y RMSE

## Autor

Randy Bonucci — Proyecto educativo, Bootcamp Data & IA

## Licencia

MIT
