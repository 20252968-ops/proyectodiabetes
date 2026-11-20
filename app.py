"""Grupo 08 — Inferencia de diabetes con un modelo de Orange guardado.

Python 3.11. No hay entrenamiento: el archivo .pkcls se carga una sola vez.
IMPORTANTE: confiar solamente en el archivo .pkcls exportado por el grupo.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from Orange.base import Model
from Orange.data import ContinuousVariable, DiscreteVariable, Table

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "modelos" / "Diabetes.pkcls"

st.set_page_config(page_title="Clasificación de diabetes | Grupo 08", layout="centered")
st.title("Clasificación de presencia de diabetes")
st.caption("IA Aplicada · Grupo 08 · Modelo Random Forest entrenado en Orange")
st.warning(
    "Demostración académica. La clasificación no constituye un diagnóstico médico "
    "ni debe utilizarse para tomar decisiones clínicas. No introduzcas datos personales reales."
)

@st.cache_resource(show_spinner="Cargando el modelo de Orange…")
def load_model(path: str):
    # El formato pickle puede ejecutar código durante su carga.
    # Nunca aceptar un archivo .pkcls proporcionado por el visitante.
    with open(path, "rb") as stream:
        model = pickle.load(stream)
    if not callable(model) or not hasattr(model, "domain"):
        raise TypeError("El archivo no contiene un modelo de Orange compatible.")
    return model

if not MODEL_PATH.is_file():
    st.error("No se encuentra `modelos/Diabetes.pkcls`.")
    st.info("Copia el modelo exportado desde Orange en la carpeta `modelos` y reinicia la app.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("El modelo no pudo cargarse en este entorno.")
    st.code(f"{type(error).__name__}: {error}")
    st.info("Verifica las versiones de Python, Orange3 y scikit-learn usadas al exportarlo.")
    st.stop()

# El Random Forest exportado guarda variables transformadas (indicadores del IMC)
# en model.domain. Para entrada humana usamos las variables originales.
input_domain = getattr(model, "original_domain", None) or model.domain
output_domain = model.domain
class_variable = output_domain.class_var
if not isinstance(class_variable, DiscreteVariable):
    st.error("El modelo no contiene una variable objetivo categórica válida.")
    st.stop()
if not input_domain.attributes:
    st.error("El modelo no declara predictores originales.")
    st.stop()

labels = {
    "Glucosa": "Glucosa",
    "Presion_Arterial": "Presión arterial",
    "Insulina": "Insulina",
    "Descripcion_IMC": "Descripción del IMC",
    "Funcion_Diabetes_Pedigree": "Función de diabetes pedigree",
    "Edad": "Edad (años)",
    "peso": "Peso",
    "Altura": "Altura",
}

st.subheader("Características del registro")
st.caption(
    "Los campos proceden del dominio ORIGINAL del modelo de Orange. "
    "Los valores faltantes se representarán como desconocidos."
)

values = []
with st.form("formulario_clasificacion"):
    for index, attribute in enumerate(input_domain.attributes):
        label = labels.get(attribute.name, attribute.name)
        widget_key = f"field_{index}_{attribute.name}"
        if isinstance(attribute, ContinuousVariable):
            value = st.number_input(
                label,
                min_value=None,
                value=None,
                format="%.4f",
                placeholder="Sin dato",
                key=widget_key,
                help="Dejar vacío si se desconoce la medición.",
            )
            values.append(np.nan if value is None else float(value))
        elif isinstance(attribute, DiscreteVariable):
            options = ["(Sin dato)"] + list(attribute.values)
            chosen = st.selectbox(label, options, key=widget_key)
            values.append(np.nan if chosen == "(Sin dato)" else float(attribute.values.index(chosen)))
        else:
            st.error(f"Variable no admitida: {attribute.name} ({type(attribute).__name__}).")
            st.stop()

    submit = st.form_submit_button("Clasificar registro", type="primary", use_container_width=True)

if submit:
    if all(np.isnan(v) for v in values):
        st.warning("Ingresa al menos una característica antes de clasificar.")
        st.stop()
    try:
        # Orange exige respetar TODAS las columnas del dominio: predictores X,
        # objetivo Y y metadatos M. Para inferencia, Y y M son desconocidos.
        # Por ejemplo, Nombre_Paciente es un metadato y NO es un predictor.
        observation = Table.from_numpy(
            input_domain,
            X=np.asarray([values], dtype=float),
            Y=np.full((1, len(input_domain.class_vars)), np.nan, dtype=float),
            metas=np.full((1, len(input_domain.metas)), np.nan, dtype=object),
        )
        response = model(observation, Model.ValueProbs)
        if not isinstance(response, tuple) or len(response) != 2:
            raise ValueError("El modelo no devolvió clases y probabilidades.")
        predictions, probabilities = response
        prediction_idx = int(np.asarray(predictions).reshape(-1)[0])
        label = class_variable.values[prediction_idx]
        st.subheader("Resultado")
        st.metric("Clasificación estimada", label)
        probs = np.asarray(probabilities, dtype=float).reshape(-1, len(class_variable.values))[0]
        if np.all(np.isfinite(probs)):
            st.write("**Probabilidades estimadas**")
            for name, probability in zip(class_variable.values, probs):
                st.write(f"{name}: {float(probability):.1%}")
                st.progress(float(np.clip(probability, 0, 1)))
        else:
            st.info("Este modelo no suministró probabilidades válidas.")
    except Exception as error:
        st.error("No se pudo generar la clasificación.")
        st.code(f"{type(error).__name__}: {error}")
        st.info("Compara los campos y el resultado con el widget Predictions de Orange.")

st.divider()
st.caption("La aplicación no entrena el modelo ni guarda los valores del formulario.")
