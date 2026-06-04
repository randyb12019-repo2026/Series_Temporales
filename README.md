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

## Autor

Randy Bonucci — Proyecto educativo, Bootcamp Data & IA

## Licencia

MIT
