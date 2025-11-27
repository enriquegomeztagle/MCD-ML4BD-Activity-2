import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea", page_icon="")
st.title("Aprendizaje en línea con River (Step-by-step desde GCS)")

st.markdown(
    """
Este panel replica **exactamente** la lógica original del código del estudiante,
pero ahora permite procesar **un archivo por clic**, en lugar de procesar todo el bucket.
"""
)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: {destination_blob}")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")


def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        return None

    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None


# =========================================================
# PARÁMETROS
# =========================================================
bucket_name = st.text_input("Bucket de GCS:", "bucket-grandesdatos-151025")
prefix = st.text_input("Prefijo/carpeta:", "tlc_yellow_trips_2022/")
limite = st.number_input("Filas a procesar por archivo:", value=1000, step=100)

MODEL_PATH = "models/model_incremental.pkl"

# =========================================================
# INICIALIZAR MODELO
# =========================================================
if "model" not in st.session_state:
    # ALWAYS start with fresh model (don't load from GCS)
    # Use MinMaxScaler instead of StandardScaler for better numerical stability
    # when features have low initial variance (e.g., weekend-only data in first file)
    model = preprocessing.MinMaxScaler() | linear_model.LinearRegression()

    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []

    # lista de archivos del bucket e índice actual
    st.session_state.blobs = None
    st.session_state.index = 0

model = st.session_state.model
metric = st.session_state.metric


# =========================================================
# FEATURE ENGINEERING (idéntico al estudiante)
# =========================================================
def _parse_time_fields(row):
    if "pickup_hour" in row and pd.notna(row["pickup_hour"]):
        try:
            hour = int(pd.to_numeric(row["pickup_hour"], errors="coerce"))
            return None, max(0, min(hour, 23))
        except:
            pass

    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce", utc=False)
            if pd.notna(dt):
                return dt, int(dt.hour)
    return None, 0


def _extract_x(row):
    dist = float(pd.to_numeric(row.get("trip_distance", 0), errors="coerce") or 0)
    psg = float(pd.to_numeric(row.get("passenger_count", 0), errors="coerce") or 0)

    dt, hour = _parse_time_fields(row)
    dow = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    weekend = 1.0 if dow >= 5 else 0.0

    # Using all 6 features with MinMaxScaler (more stable than StandardScaler)
    return {
        "dist": dist,
        "log_dist": float(np.log1p(max(dist, 0))),
        "pass": psg,
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": weekend,
    }


def _valid_target(v):
    y = pd.to_numeric(v, errors="coerce")
    if pd.isna(y):
        return None
    return float(y)


# =========================================================
# PROCESAR UN SOLO ARCHIVO (MISMA LÓGICA DEL ESTUDIANTE)
# =========================================================
def process_single_blob(bucket_name, blob_name, limite=1000, chunksize=500):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    try:
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)
        count = 0

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):

            if not {"trip_distance", "passenger_count", "fare_amount"}.issubset(
                chunk.columns
            ):
                continue

            for col in ["trip_distance", "passenger_count", "fare_amount"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            chunk = chunk[
                chunk["fare_amount"].between(2, 200)
                & chunk["trip_distance"].between(0.1, 50)
                & chunk["passenger_count"].between(1, 6)
            ]

            for _, row in chunk.iterrows():
                if count >= limite:
                    break

                y = _valid_target(row["fare_amount"])
                if y is None:
                    continue

                x = _extract_x(row)

                pred = model.predict_one(x)
                model.learn_one(x, y)
                metric.update(y, pred)

                count += 1

            # Exit outer loop if limit reached
            if count >= limite:
                break

    except Exception as e:
        st.warning(f"Error en {blob_name}: {e}")
        return None

    return metric.get()


# =========================================================
# BOTÓN: PROCESAR SIGUIENTE ARCHIVO
# =========================================================
if st.button("Procesar siguiente archivo"):

    if st.session_state.blobs is None:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        st.session_state.blobs = list(bucket.list_blobs(prefix=prefix))
        st.session_state.index = 0
        st.info(f"Se encontraron {len(st.session_state.blobs)} archivos.")

    blobs = st.session_state.blobs
    idx = st.session_state.index

    if idx >= len(blobs):
        st.success("Todos los archivos ya fueron procesados.")
    else:
        blob = blobs[idx]
        short = blob.name.split("/")[-1]
        st.write(f"Procesando {idx+1}/{len(blobs)}: `{short}`")

        score = process_single_blob(bucket_name, blob.name, int(limite))

        if score is not None:
            st.session_state.history.append(score)
            st.write(f"{blob.name} — R² acumulado: **{score:.3f}**")
            save_model_to_gcs(model, bucket_name, MODEL_PATH)
            print(f"[R2 INFO] Processed {blob.name} - Current R2: {score}")

        st.session_state.index += 1

# =========================================================
# ESTADO FINAL
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{metric.get():.3f}**")
st.write(
    f"Archivos procesados: **{st.session_state.index}/{len(st.session_state.blobs) if st.session_state.blobs else 0}**"
)

if st.session_state.history:
    st.line_chart(st.session_state.history)

# =========================================================
# BOTÓN DE RESET
# =========================================================
if st.button("🔄 Reiniciar modelo y comenzar de nuevo"):
    st.session_state.clear()
    st.rerun()

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)_221125")
