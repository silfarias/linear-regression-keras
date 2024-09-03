# Modelo de Regresion Lineal con Keras

En este trabajo hemos implementado un modelo de regresion lineal utilizando la libreria Keras basado en un conjunto de datos que incluye altura y peso de personas. Este modelo permite encontrar la relación entre la altura y el peso de una persona, realizar predicciones y visualizar resultados.


## Requisitos

- Python 3.x
- pandas
- numpy
- matplotlib
- keras (con TensorFlow)

## Pasos
1. Clonar este repositorio

```bash
git clone https://github.com/silfarias/linear-regression-keras.git
```

2. Moverse al directorio

```bash
cd linear-regression-keras
```

3. Instalar dependencias 
```bash
pip install pandas numpy matplotlib tensorflow keras
```

4. Ejecutar el script
```bash
python main.py
```

### Explicacion

Desarrollamos un modelo de regresión lineal simple para analizar la relación entre la altura y el peso de un conjunto de datos. El modelo fue implementado utilizando Keras, con un enfoque en la normalización de los datos antes del entrenamiento para mejorar la precisión y estabilidad del modelo.

### Resultados Obtenidos:

Tras entrenar el modelo con 10000 épocas, obtuvimos un peso (w) y un sesgo (b) que definen la recta de regresión. La recta de regresión se ajusta bien a los datos, lo que confirma que existe una relación lineal significativa entre la altura y el peso en el conjunto de datos proporcionado. Visualizamos esta relación con gráficos que muestran tanto la evolución del error cuadrático medio (ECM) como la comparación entre los datos originales y la recta de regresión.
Los graficos pueden ser visualizados en la carpeta **graficas**

### Interpretacion:

Los resultados del modelo muestran que, con la información de la altura de una persona, es posible predecir su peso con un cierto nivel de precisión. Esto se debe a la correlación lineal positiva entre ambas variables