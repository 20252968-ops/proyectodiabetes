# Grupo 08 — Clasificación de presencia de diabetes

Aplicación académica con **Python 3.11**, **Streamlit** y un modelo **Random Forest ya entrenado en Orange Data Mining**. La aplicación no vuelve a entrenar el modelo.

> La clasificación es experimental y **no representa un diagnóstico médico**.

## Contenido

```text
proyecto-diabetes/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── modelos/
    └── Diabetes.pkcls  ← colocar el archivo entregado por el grupo
```

## 1. Modelo exportado

El archivo proporcionado `Diabetes.pkcls` ha sido inspeccionado **sin deserializarlo**: su contenido se identifica como un modelo de **Orange RandomForestClassifier** entrenado con **100 árboles**, clases `Ausencia` y `Presencia`, y registra **scikit-learn 1.5.2**. Incluye columnas transformadas de `Descripcion_IMC` y una referencia al dominio original de entrada. La app obtiene sus campos desde ese dominio original.

**Precaución de privacidad:** el archivo serializado incluye una tabla de datos de entrenamiento y nombres de pacientes. **No publiques este `.pkcls` en un repositorio GitHub público ni lo despliegues sin eliminar estos datos del modelo exportado**. Un `.gitignore` excluye el archivo del control de versiones, pero no elimina archivos ya publicados previamente. Verifica si los nombres son ficticios; aun en ese caso, revisa la política del curso y el origen de los datos. La ruta de despliegue público requiere un modelo exportado sin datos de personas.

**Seguridad:** archivos pickle (`.pkcls`) pueden ejecutar código al cargarse. La app solo carga el archivo local del proyecto. Utiliza exclusivamente archivos confiables exportados por tu grupo; no acepta archivos subidos por los visitantes.

## 2. Preparar el entorno local de Windows

Instala **Python 3.11** y ejecuta en PowerShell, desde la carpeta del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La versión de **scikit-learn se fijó en 1.5.2** porque está registrada en el modelo. La versión de **Orange3 3.41.0 es provisional**: no se pudo confirmar la versión exacta con la que se exportó el modelo. Los modelos serializados pueden no funcionar al cambiar versiones. Para comprobar la versión real en la máquina donde se exportó:

```powershell
python -m pip show Orange3 scikit-learn numpy scipy
```

Si es diferente, ajusta el `requirements.txt` y reconstruye el entorno virtual sin reentrenar el modelo. Si el modelo original depende de bibliotecas compiladas incompatibles, la carga puede fallar: eso debe resolverse usando las versiones originales.

## 3. Colocar el modelo y ejecutar

Copia `Diabetes.pkcls` dentro de la carpeta `modelos/` y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre la URL local indicada por Streamlit (usualmente `http://localhost:8501`).

La aplicación lee el modelo al iniciarse, muestra los campos de entrada a partir del dominio original e invoca su método de inferencia. No incluye `fit()`, no importa ni entrena con el dataset y no modifica el modelo guardado.

## 4. Prueba funcional requerida

1. Abre la aplicación local y verifica que cargue sin excepción.
2. Introduce una observación de prueba sin datos identificatorios.
3. Comprueba que aparezcan `Presencia` o `Ausencia` y, cuando sea posible, sus probabilidades.
4. Repite la misma observación en el widget **Predictions** de Orange y compara las salidas.
5. Comprueba qué reglas de imputación y codificación forman parte del modelo exportado. No supongas que la app corrige automáticamente valores de glucosa iguales a cero.

La sintaxis de `app.py` está comprobada, pero **no se ha cargado ni ejecutado el `.pkcls` en esta sesión**: la carga de pickle no se realizó por seguridad. Por ello, las predicciones reales y la compatibilidad exacta siguen pendientes de validación local.

## 5. GitHub y Streamlit Community Cloud

Los tres archivos de código y documentación pueden publicarse en GitHub. **No subas el `Diabetes.pkcls` actual a GitHub público**, porque contiene datos de entrenamiento con nombres. Para realizar un despliegue público debes preparar y verificar un modelo sin registros ni nombres incrustados; el `.pkcls` proporcionado sirve solo para ejecución local bajo control del grupo.

Una vez que dispongas de un modelo apto para publicación, Streamlit Community Cloud permite desplegar el repositorio señalando `app.py` como entrada y el entorno Python compatible. Es necesario verificar que la instalación de Orange3, scikit-learn y sus bibliotecas compiladas funcione en ese servicio. No se ha probado ese despliegue.

## 6. Autoría académica

Declara el apoyo de IA en el código, según lo dispuesto por el curso. El workflow y el informe final deben respetar las condiciones de autoría y documentación de la asignatura.
