import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

def lectura_csv(file):
    try:
        dataFrame = pd.read_csv(file)
        return dataFrame
    except FileNotFoundError:
        print(f'No se encontro el archivo {file}')
        return None
        

def normalizar_datos(x, y):
    # utilizamos la tecnica min-max scaling para normalizar los datos
    # asi nos aseguramos que los valores esten entre 0 y 1
    n_x = (x - x.min()) / (x.max() - x.min())
    n_y = (y - y.min()) / (y.max() - y.min())
    return n_x, n_y


def generation_model():
    np.random.seed(2) # replica los resultados en distintas computadores
    modelo = Sequential() # crea una caja vacia al que iremos añadiendo elementos
    
    # tamaño de los datos de entrada y salida
    input_dim = 1 # una dimension
    output_dim = 1
    
    # capa de entrada, capa de salida y la funcion de activacion
    capa = Dense(output_dim, input_dim=input_dim, activation='linear') 
    modelo.add(capa)
    
    # creamos una instancia del gradiente descendente con una tasa de aprendizaje
    sdg = SGD(learning_rate=0.0004)
    modelo.compile(loss='mse', optimizer=sdg) # funcion de error: Mean Squared Error
    modelo.summary() # mostramos resumen del modelo
    return modelo
    

def entrenamiento_modelo(modelo, n_x, n_y):

    num_epochs = 1000 # numero de ciclos de entrenamiento
    batch_size = n_x.shape[0] # el tamaño de datos 
    
    history = modelo.fit(n_x, n_y, epochs=num_epochs, batch_size=batch_size, verbose=1) # resultados del entrenamiento, comportamiento de perdida 
    
    capa = modelo.layers[0]
    
    # parametros del modelo para minimizar la perdida
    w, b = capa.get_weights() # peso y sesgo
    print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0], b[0]))
    
    return history, w[0][0], b[0]


# mostramos como evoluciona el ECM
def grafico_ecm(history):
    # plt.subplot(1, 2, 1)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], 'b-', label='ECM')
    plt.xlabel('Épocas')
    plt.ylabel('ECM')
    plt.title('ECM vs. Épocas')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def grafico_regresion(data, w, b):
    x = data['Altura'].values
    y = data['Peso'].values

    # normalizamos x (altura)
    n_x = (x - x.min()) / (x.max() - x.min())

    # predicción utilizando el modelo con datos normalizados
    y_pred_normalizado = w * n_x + b

    # volvemos a escalar las predicciones a la escala original de y (peso)
    y_pred = y_pred_normalizado * (y.max() - y.min()) + y.min()

    plt.scatter(x, y, label='Datos originales', color='blue')
    plt.plot(x, y_pred, label='Recta de Regresión', color='red')
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.title('Recta de Regresión vs. Datos Originales')
    plt.legend()
    plt.show()


def prediccion(modelo, altura_cm, data):
    alt_min = data['Altura'].min()
    alt_max = data['Altura'].max()

    alt_norm = (altura_cm - alt_min) / (alt_max - alt_min)
    
    y_pred = modelo.predict(np.array([alt_norm]))
    
    peso_min = data['Peso'].min()
    peso_max = data['Peso'].max()
    peso = y_pred[0][0] * (peso_max - peso_min) + peso_min
    
    print(f'El peso para una persona de {altura_cm} cm es de {peso:.2f} kg')
    return peso


def main():
    data = lectura_csv('altura_peso.csv')

    # obtenemos los valores originales de altura y peso
    x = data['Altura'].values
    y = data['Peso'].values

    n_x, n_y = normalizar_datos(x, y)

    # sustituimos los valores originales por los normalizados
    data['Altura'] = n_x
    data['Peso'] = n_y

    # genera el modelo y entrena
    modelo = generation_model()
    history, w, b = entrenamiento_modelo(modelo, n_x, n_y)

    grafico_ecm(history)
    grafico_regresion(data, w, b)
    
    altura = 170
    prediccion(modelo, altura, data)
    
if __name__ == '__main__':
    main()