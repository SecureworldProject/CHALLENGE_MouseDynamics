import numpy as np 
import tensorflow as tf
from laberintoCondiff import laberinto

from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

#--------------------------------------------------------------------------
def plot_confusion_matrix(cm):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

logdir = "logs/plots5/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')




def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(X_trainf)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(Y_train, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
#------------------------------------------------------------------------------------------------



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
                aux=np.zeros(shape=[ventana,2])
                aux[0:len(array)]=array
                X_array.append(aux)
            except:
                aux=np.zeros(shape=[ventana])
                aux[0:len(array)]=array
                X_array.append(aux)
        return X_array

#leyemos los datos
#-------------------------------------------------------------------------------------------------
#for x in range(1, 60, 5):
print('---------------------------------------------------------------------------------------------------------------')
#-------------------------------------------------------------------------------------------------------------
ventana=900



#metodo1 (x,Y)
#metodo2 (deltaX,deltay)
#,etodo3 (hipo)
metodo=2




#-------------------------------------------------------------------------------------------------------------

print(ventana)
def leerDatos(metodo):
    Y_trainf=[]
    X_trainf=[]
    
    X_train=np.load("datos"+str(metodo)+"/1.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/6.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/7.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/8.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/9.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/10.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/11.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/12.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/13.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/14.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/15.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/16.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/17.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/18.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/19.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])


    X_train=np.load("datos"+str(metodo)+"/20.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/21.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/22.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/23.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/24.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/25.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/26.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/27.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/28.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/29.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/30.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    


    """

    #-------------------------------------------------------------------------------------------------
    """
    X_train=np.load("datos"+str(metodo)+"/1001.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1002.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1003.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1004.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1005.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1006.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1007.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1008.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1009.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/1010.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    
    X_train=np.load("datos"+str(metodo)+"/1011.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1012.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1013.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1014.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1015.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1016.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1017.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1018.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1019.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1020.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1021.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1022.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1023.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1024.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1025.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1026.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1027.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1028.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1029.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1030.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/1031.npy")
    aux1,aux2=arrayentero(X_train,ventana,1)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    
    #---------------------------------------------------------------------------------------------


    X_train=np.load("datos"+str(metodo)+"/2001.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])


    X_train=np.load("datos"+str(metodo)+"/2002.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2003.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2004.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2005.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2006.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2007.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2008.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2009.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2010.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2011.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2012.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2013.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2014.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2015.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2016.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2017.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2018.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2019.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2020.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2021.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2022.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2023.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2024.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2025.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2026.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2027.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2028.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2029.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2030.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/2031.npy")
    aux1,aux2=arrayentero(X_train,ventana,2)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    #---------------------------------------------------------------------
    
    X_train=np.load("datos"+str(metodo)+"/3001.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])


    X_train=np.load("datos"+str(metodo)+"/3002.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3003.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3004.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3005.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3006.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3007.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3008.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3009.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3010.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3011.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3012.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3013.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3014.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3015.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3016.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3017.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3018.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3019.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3020.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3021.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3022.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3023.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3024.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3025.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3026.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3027.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3028.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3029.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/3030.npy")
    aux1,aux2=arrayentero(X_train,ventana,3)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    """
    #------------------------------------------------------------------------------------
    """
    X_train=np.load("datos"+str(metodo)+"/4001.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])


    X_train=np.load("datos"+str(metodo)+"/4002.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4004.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4004.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4005.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4006.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4007.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4008.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4009.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4010.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4011.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4012.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4014.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4014.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4015.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4016.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4017.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4018.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4019.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4020.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4021.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4022.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4024.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4024.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4025.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4026.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4027.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4028.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4029.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/4030.npy")
    aux1,aux2=arrayentero(X_train,ventana,4)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    #------------------------------------------------------------------------
    X_train=np.load("datos"+str(metodo)+"/5001.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])


    X_train=np.load("datos"+str(metodo)+"/5002.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5004.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5004.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5005.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5006.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5007.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5008.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5009.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5010.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5011.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5012.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5014.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5014.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5015.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5016.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5017.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5018.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5019.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5020.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5021.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5022.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5024.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5024.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5025.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5026.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5027.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5028.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5029.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/5030.npy")
    aux1,aux2=arrayentero(X_train,ventana,5)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    print(np.max(X_trainf))
    print(np.min(X_trainf))
    #capturamos los datos del usuario habitual 
    #-----------------------------------------------------------------------
    """
    datos=laberinto()
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    datos=laberinto()
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    """

    print(X_train)

    X_trainf=np.array(X_trainf)
    Y_train=np.array(Y_trainf)
    print(X_trainf)
    return X_trainf,Y_train
#----------------------------------------------------------------------------------------------------------------------
X_trainf,Y_train=leerDatos(metodo)
#-----------------------------------------------------------------------------------------------
#cambiar
#X_trainf=X_trainf/200
#X_trainf=X_trainf/200




Xtrain,Ytrain=leerDatos(3)
#Xtrain=Xtrain/200

X_trainf=np.concatenate([X_trainf, np.reshape(Xtrain, (len(Xtrain),ventana,1))], axis=2)
print(len(X_trainf))
print(len(Ytrain))
"""
for i in range(1,30):

    datos1,datos2=laberinto()
    aux1,aux2=arrayentero(datos1,ventana,0)
    aux3,aux4=arrayentero(datos2,ventana,0)
    prueba=np.concatenate([aux3, np.reshape(aux1, (1,ventana,1))], axis=2)
    X_trainf=np.concatenate([prueba,X_trainf],axis=0)
    Y_train=np.concatenate([aux2,Y_train],axis=0)
    print(len(X_trainf))
    print(len(Y_train))

"""

#print(X_trainf)
print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
print((X_trainf))
print(Ytrain)

#metodo=5
#-----------------------------------------------------------------------------------------------
if(metodo==1):
    X_trainf=X_trainf/640
if(metodo==2):
    X_trainf=X_trainf/200
if(metodo==3):
    #pensar
    X_trainf=X_trainf/200


    
#----------------------------------------------------------------------------------------------------------------------
"""
print(np.max(X_trainf))
print(np.min(X_trainf))
print(X_trainf )
"""


#print(X_trainf)
# [?,250,2,1], [3,3,1,25].
#modelo de deep learning
model=tf.keras.Sequential([
    tf.keras.layers.Reshape((ventana,3,1),input_shape=np.shape(X_trainf [0])),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(6,2), activation='relu'),
    tf.keras.layers.MaxPooling2D((4, 2)),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4,1), activation='tanh'),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(6,activation=tf.nn.softmax)

])


#compilamos el modelo 
model.compile(optimizer='adam'
                , loss='sparse_categorical_crossentropy'
                , metrics=["accuracy"]
                )
X_trainf=tf.random.shuffle(X_trainf, seed=1234)
Y_train=tf.random.shuffle(Y_train, seed=1234)
#lo entrenamos
model.fit(X_trainf,Y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.1
            #callbacks=[tensorboard_callback, cm_callback],
            )


print('---------------------------------------------------------------------------------------------------------------')


model.save('path_to_my_model.h5')

def datoshabitual(metodo):
    Y_trainf=[]
    X_trainf=[]

    X_train=np.load("datos"+str(metodo)+"/1.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/2.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/3.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/4.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/5.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/6.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/7.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/8.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/9.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/10.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/11.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/12.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/13.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/14.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/15.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/16.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/17.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/18.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])

    X_train=np.load("datos"+str(metodo)+"/19.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])


    X_train=np.load("datos"+str(metodo)+"/20.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/21.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/22.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/23.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/24.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/25.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/26.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/27.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/28.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/29.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_train=np.load("datos"+str(metodo)+"/30.npy")
    aux1,aux2=arrayentero(X_train,ventana,0)
    #aux=np.zeros(shape=[4000,2])
    #aux[0:len(X_train)]=X_train
    X_trainf.append(aux1[0])
    Y_trainf.append(aux2[0])
    X_trainf=np.array(X_trainf)
    Y_train=np.array(Y_trainf)
    print(X_trainf)
    return X_trainf,Y_train



X_trainf,Y_train=datoshabitual(metodo)

Xtrain,Ytrain=datoshabitual(3)

X_trainf=np.concatenate([X_trainf, np.reshape(Xtrain, (len(Xtrain),ventana,1))], axis=2)



if(metodo==1):
    X_trainf=X_trainf/640
if(metodo==2):
    X_trainf=X_trainf/200
if(metodo==3):
    #pensar
    X_trainf=X_trainf/200

predict=model.predict(X_trainf)
#np.bincount(
cad=np.bincount(np.argmax(predict, axis=1))
cad=np.where(cad>0)
np.save('categorias.npy',cad)

"""
datos, datosxy=laberinto()
datos=arrayentero(datos,ventana)

datos=np.array(datos)
datos=datos/100

predict=model.predict(datos)
print(predict)
cad=np.bincount(np.argmax(predict, axis=1)).argmax()
print(cad)
"""