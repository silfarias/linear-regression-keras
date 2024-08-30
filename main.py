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
    
    # print("revisando datos:")
    # print(f"min altura: {x.min()}, max altura: {x.max()}")
    # print(f"min Peso: {y.min()}, max peso: {y.max()}")
    
    
    num_epochs = 1000 # numero de ciclos de entrenamiento
    batch_size = x.shape[0] # el tamaño de datos 
    
    history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1) # resultados del entrenamiento, comportamiento de perdida 
    
    capa = modelo.layers[0]
    
    # parametros del modelo para minimizar la perdida
    w, b = capa.get_weights() # peso y sesgo
    print('Parámetros: w = {:.4f}, b = {:.4f}'.format(w[0][0], b[0]))
    
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
    
    y_prediccion = w * x + b
    
    plt.scatter(x, y, label='Datos originales', color='blue')
    plt.plot(x, y_prediccion, label='Recta de Regresión', color='red')
    plt.xlabel('Altura (Normalizada)')
    plt.ylabel('Peso (Normalizado)')
    plt.title('Recta de Regresión vs. Datos Originales')
    plt.legend()
    plt.show()


def prediccion(modelo, altura_cm, data):
    alt_max = data['Altura'].max()
    alt_norm = altura_cm / alt_max
    
    y_pred = modelo.predict(np.array([alt_norm]))
    
    
    peso_max = data['Peso'].max()
    peso = y_pred[0][0] * peso_max
    
    print(f'El peso para una persona de {altura_cm} cm es de {peso:.2f} kg')
    return peso


def main():
    data = lectura_csv('altura_peso.csv')
    data = normalizar_datos(data)
    modelo = generation_model()
    history, w, b = entrenamiento_modelo(modelo, data)
    grafico_ecm(history)
    grafico_regresion(data, w, b)
    
    altura = 170
    prediccion(modelo, altura, data)
    
if __name__ == '__main__':
    main()