import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

def lectura_csv(file):
    dataFrame = pd.read_csv(file)
    return dataFrame


def normalizar_datos(data):
    data['Altura'] = data['Altura'] / data['Altura'].max() # dividimos cada valor por el maximo
    data['Peso'] = data['Peso'] / data['Peso'].max() # asi aseguramos que los valores esten entre 0 y 1
    return data


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
    sdg = SGD(learning_rate=0.0001)
    
    modelo.compile(loss='mse', optimizer=sdg) # funcion de error: Mean Squared Error

    modelo.summary() # mostramos resumen del modelo
    return modelo
    

def entrenamiento_modelo(modelo, data):
    
    x = data['Altura'].values
    y = data['Peso'].values
    
    print("Revisando los datos:")
    print(f"Min Altura: {x.min()}, Max Altura: {x.max()}")
    print(f"Min Peso: {y.min()}, Max Peso: {y.max()}")
    
    
    num_epochs = 50 # numero de ciclos de entrenamiento
    batch_size = x.shape[0] # el tamaño de datos 
    
    history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1) # resultados del entrenamiento
    
    capa = modelo.layers[0]
    w, b = capa.get_weights()
    print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0], b[0]))
    

def graficos(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('ECM')
    plt.title('ECM vs. epochs')



data = lectura_csv('altura_peso.csv')
data = normalizar_datos(data)
modelo = generation_model()
entrenamiento_modelo(modelo, data)