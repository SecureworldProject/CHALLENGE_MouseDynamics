#importamos todas las librerias correspondientes
import lock
import os
import numpy as np
from tensorflow import keras
from laberintoCondiff import laberinto


def arrayentero(array,dim, num=None):
    Y_array=[]
    X_array=[]
    #trasformamos el array en float
    array=np.array(array,dtype='float16')
    #guardamos la dimension de la ventana en una variable
    ventana=dim
    if num is not None:
        #if array[0]==0:
        #   array=np.delete(array,0)
        #return [[np.median(array),np.mean(array),np.std(array)]], [num]
        #recorremos todos los elementos del array hasta N-longitud de la ventana y lo guardamos en un nuevo array y generamos el array de etiquetas con el num
        for i in range(0,len(array)-ventana):
            X_array.append(array[i:ventana+i])
            Y_array.append([num])
        if len(X_array)==0:
            try:
                aux=np.zeros(shape=[ventana,2])
                aux[0:len(array)]=array
                X_array.append(aux)
            except:
                aux=np.zeros(shape=[ventana])
                aux[0:len(array)]=array
                X_array.append(aux)
            Y_array.append([num])
        Y_array=np.array(Y_array)
        return X_array,Y_array
    else:
        #recorremos todos los elementos del array hasta N-longitud de la ventana y lo guardamos en un nuevo array
        for i in range(0,len(array)-ventana):
            X_array.append(array[i:ventana+i,:])
        if len(X_array)==0:
            try:
                aux=np.zeros(shape=[ventana,3])
                aux[0:len(array)]=array
                X_array.append(aux)
            except:
                aux=np.zeros(shape=[ventana])
                aux[0:len(array)]=array
                X_array.append(aux)
        return X_array


props_dict = {}
DEBUG_MODE = True

# funcion init devuelve un 0 que es valido ya que suponemos que el usuario va a tener siempre 8un teclado con el que pueda escribir
def init(props):
    global props_dict
    print("Python: starting challenge init()")
    #cargamos el json que le pasemos y lo guardamos en la variable global
    props_dict = props
    return 0



def executeChallenge():
    print("Python: starting executeChallenge()")
    #comprobamos las variables de entorno y cogemos el de SECUREMIRROR_CAPTURES
    dataPath = os.environ['SECUREMIRROR_CAPTURES']
    print ("storage folder is :",dataPath)
    #abrimos lock
    lock.lockIN("keystroke")
    metodo=2
    #ejecutamos el codigo de captura de datos 
    if metodo==1:
        datos=laberinto()
        aux=np.zeros(shape=[4000,2])
        aux[0:len(datos)]=datos
        aux=aux/640
        datos=np.array([aux])
    else:
        ventana=900
        datos=laberinto()
        datos=arrayentero(datos,ventana)
        """
        aux=np.zeros(shape=[4000,2])
        aux[0:len(datos)]=datos"""
        datos=np.array(datos)
        datos=datos/640
        
    #seleccionamos la dimension de la ventana
    #tratamos los datos
    #cargamos el modelo
    #############################
    #cambiar la ruta si se pasa por el json
    url=props_dict["url"]
    new_model = keras.models.load_model(url+'path_to_my_model.h5')
    #cerramos el lock
    lock.lockOUT("mouse_Dinamics")
    #predecimos la categoria de los nuevo datos

    new_predictions = new_model.predict(datos)
    print(new_predictions)
    print(np.argmax(new_predictions, axis=1))
    cad=np.argmax(new_predictions, axis=1)
    categorias=np.load("categorias.npy")
    cat=np.where(cad==categorias[0])
    if len(cat)!=0:
        cad=0
    #y generamos el resultado
    cad="%d"%(cad)
    key = bytes(cad,'utf-8')
    key_size = len(key)
    result = (key, key_size)
    print("result:", result)
    return result


# esta parte del codigo no se ejecuta a no ser que sea llamada desde linea de comandos
if __name__ == "__main__":
    midict = {}
    init(midict)
    executeChallenge()
