from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from predicterapp.Utils import *
from predicterapp.PreProcesamiento import desNormalizar

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def regresionPolinomial(nombreDatos, datosAdicionales, ventana, diasAPredecir, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest):
    '''Leemeos los datos almacenados'''
    ruta = '\predicterapp\static\predicterapp\myDates\dataframe\\' + nombreDatos + '.infer';
    datosInferClass = pd.read_pickle(BASE_DIR + ruta)
    
    datosArrayClass = np.load(BASE_DIR + '\\predicterapp\\static\\predicterapp\\myDates\\narray\\' + nombreDatos + '.npy')
    
    '''Creamos la matriz con todos los distintos conjuntos de datos seleccionados en el formulario'''
    matrizTrain, matrizTest = creacionMatrizDeRegresion(datosArrayClass, datosInferClass, datosAdicionales, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest, ventana)
    
    print("Matriz train", matrizTrain.shape)
    print("Matriz test", matrizTest.shape)
    
    '''Dividimos nuestras matrices de entrenamiento y pruebas para poder entrenar el algoritmo'''
    y_train = matrizTrain[:,matrizTrain[0].size-1]
    X_train = np.delete(matrizTrain, matrizTrain[0].size-1, 1)
    y_test = matrizTest[:,matrizTest[0].size-1]
    X_test = np.delete(matrizTest, matrizTest[0].size-1, 1)
    
    '''Declaramos nuestro algoritmo de regresion y lo entrenamos con el conjunto de entrenamiento'''
    from sklearn.kernel_ridge import KernelRidge
    #clf = KernelRidge(kernel="polynomial")
    from sklearn import linear_model
    clf = seleccionarMejorAlgoritmoRegresion(X_train, y_train, X_test, y_test)
    
    clf.fit(X_train, y_train)
    
    '''Comprobamos como es de bueno nuestro algoritmo'''
    score = clf.score(X_test, y_test)
    mae = mean_absolute_error(y_test, clf.predict(X_test))
    
    '''Creamos una matriz con todo el conjunto de datos y le asignamos la ventana para poder predecir el dia de manana'''
    
    matrizCompleta = crearMatrizRegresionCompleta(datosArrayClass, datosInferClass, datosAdicionales, ventana)
    Y = matrizCompleta[:,matrizCompleta[0].size-1]
    X = np.delete(matrizCompleta, matrizCompleta[0].size-1, 1)
    vector = X[0:1, :][0]
    print("Vector predecir", vector)
    
    prediccion = creacionVectoresParaPredecir(vector, int(float(diasAPredecir)), clf)
    
    prediccion = desNormalizar(datosInferClass.tail(1).reset_index()['Close'][0], datosArrayClass[datosArrayClass.size-1], prediccion)
    
    scoreArray = []
    maeArray = []
    
    scoreArray.append(score)
    maeArray.append(mae)
    
    '''Calculo del score de los dias siguientes'''
    if int(diasAPredecir) >= 2:
        arrayBucle = (np.arange(int(diasAPredecir)-1))+1
        for aux in arrayBucle:
            matrizTrainN, matrizTestN = creacionMatrizDePrediccionDeErroresFuturosRegresion(datosArrayClass, datosInferClass, datosAdicionales, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest, ventana, aux)
            scoreDiasPosteriores, maeDiasPosteriores = calcularScoreMaeDiasAnteriores(matrizTrainN, matrizTestN, clf)
            scoreArray.append(scoreDiasPosteriores)
            maeArray.append(maeDiasPosteriores)        
    
    return scoreArray, maeArray, prediccion


def creacionMatrizDeRegresion(datosArrayClass, datosInferClass, datosAdicionales, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest, ventana):
    
    '''Creamos la matriz completa con todos los dias disponibles en el conjunto de datos'''
    vectorCompleto = vectorCompletoInvertido(datosArrayClass)
    matrizCompleta = crearMatriz(vectorCompleto, ventana)
    
    '''Creamos las matrices correspondientes al conjunto de entrenamiento y de pruebas'''
    matrizTrain, matrizTest = matrizEntrenamientoTest(matrizCompleta, datosInferClass, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest)
    
    
    if len(datosAdicionales) != 0:
        for aux in datosAdicionales:
            '''Leemos los datos'''
            ruta = '\predicterapp\static\predicterapp\myDates\dataframe\\' + aux + '.infer';
            datosInferAdicionales = pd.read_pickle(BASE_DIR + ruta)
            datosArrayAdicionales = np.load(BASE_DIR + '\\predicterapp\\static\\predicterapp\\myDates\\narray\\' + aux + '.npy')
            
            '''Creamos el vector y la matriz completa de los datos adicionales'''
            vectorCompletoDatosAdicionales = vectorCompletoInvertido(datosArrayAdicionales)
            matrizCompletaDatosAdicionales = crearMatriz(vectorCompletoDatosAdicionales, ventana)
            
            '''Creamos las matrices correspondientes a los vectores de entrenamiento y a los de pruebas'''
            matrizTrainAdicionales, matrizTestAdicionales = matrizEntrenamientoTest(matrizCompletaDatosAdicionales, datosInferAdicionales, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest)
            
            '''Nos quedamos con la X que es lo unico que nos interesa de las matrices creadas con los datos adicionales'''
            X_trainAdicionales = np.delete(matrizTrainAdicionales, matrizTrainAdicionales[0].size-1, 1)
            X_testAdicionales = np.delete(matrizTestAdicionales, matrizTestAdicionales[0].size-1, 1)
            
            '''Unimos le conjunto de datos adicionales'''
            matrizTrain = np.concatenate((X_trainAdicionales, matrizTrain), axis = 1)
            matrizTest = np.concatenate((X_testAdicionales, matrizTest), axis = 1)
    
    return matrizTrain, matrizTest

def crearMatriz(datos, ventana):
    ventana = int(ventana)
    matriz = np.zeros((datos.size, ventana+1))
    for x in range(ventana+1):
        vector = crearVector(datos, x)
        matriz[:,ventana-x] = vector[:,0]
    return matriz[ventana:vector.size :,]

def crearVector(vector, desplazamiento):
    result = vector[0:vector.size-1-desplazamiento+1]
    for aux in range(desplazamiento):
        result = np.insert(result, 0, 0)
    return result.reshape(-1, 1)

def buscarFecha(datosInfer, fecha):
    dt = pd.to_datetime(fecha)
    return datosInfer.index.get_loc(dt, method='nearest')

def vectorDatosEntreAmbasFechas(datosArray, datosInfer, fechaInicio, fechaFin):
    indiceFechaInicio = buscarFecha(datosInfer, fechaInicio)
    indiceFechaFin = buscarFecha(datosInfer,fechaFin)
    
    return datosArray[indiceFechaInicio: indiceFechaFin]

def seleccionarVectorPredecir(Y, X):
    dimension = X.shape
    vector = X[dimension[0]-1][1:]
    vector = np.insert(vector, vector.size,Y[dimension[0]-1])
    return vector

def creacionVectoresParaPredecir(datos, dias, clf):
    result = []
    valor = clf.predict([datos])
    result.append(valor)
    for aux in range(dias-1):
        datos = datos[1:]
        datos = np.insert(datos, datos.size,valor)
        valor = clf.predict([datos])
        result.append(valor)
    return result

def crearMatrizRegresionCompleta(datosArrayClass, datosInferClass, datosAdicionales, ventana):
    primeraFecha = datosInferClass.head(1).reset_index()['Date'][0]
    ultimaFecha = datosInferClass.tail(1).reset_index()['Date'][0]
     
    matrizTrain, matrizTest = creacionMatrizDeRegresion(datosArrayClass, datosInferClass, datosAdicionales, primeraFecha, ultimaFecha, primeraFecha, ultimaFecha, ventana)
     
    return matrizTrain

def vectorCompletoInvertido(datosArray):
    return np.flip(datosArray, axis=0)

def matrizEntrenamientoTest(matrizCompleta, datosInfer, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest):
    '''Localizamos los indices de las fechas pasadas en los datos de tipo infer'''
    datosInferInvertidos = datosInfer.iloc[::-1]
    indiceTrainInicio =  buscarFecha(datosInferInvertidos, fechaInicioTrain)
    indiceTrainFin =  buscarFecha(datosInferInvertidos, fechaFinTrain)
    indiceTestInicio =  buscarFecha(datosInferInvertidos, fechaInicioTest)
    indiceTestFin =  buscarFecha(datosInferInvertidos, fechaFinTest)
    
    '''Creamos las matrices de train y test'''
    matrizTrain = matrizCompleta[indiceTrainFin:indiceTrainInicio, :]
    matrizTest = matrizCompleta[indiceTestFin:indiceTestInicio, :]
    
    return matrizTrain, matrizTest
    
def seleccionarMejorAlgoritmoRegresion(X_train, y_train, X_test, y_test):
    scores = {}
    '''Importacion de todos los algoritmos que vamos a implementar'''
    from sklearn.kernel_ridge import KernelRidge
    from sklearn import linear_model
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    
    '''Declaracion de los algoritmos y entrenamiento de los algortimos'''
    '''Kernel Ridge'''
    kernelRidge = KernelRidge(kernel="polynomial")
    kernelRidge.fit(X_train, y_train)
    scoreKernelRidge = kernelRidge.score(X_test, y_test)
    scores[kernelRidge] = scoreKernelRidge
    
    '''Bayesian Ridge'''
    bayesianRidge = linear_model.BayesianRidge()
    bayesianRidge.fit(X_train, y_train)
    scoreBayesianRidge = bayesianRidge.score(X_test, y_test)
    scores[bayesianRidge] = scoreBayesianRidge
    
    '''Linear Regression'''
    linearRegression = linear_model.LinearRegression()
    linearRegression.fit(X_train, y_train)
    scoreLinearRegression = linearRegression.score(X_test, y_test)
    scores[linearRegression] = scoreLinearRegression
    
    '''Ridge Regression'''
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    scoreRidge = ridge.score(X_test, y_test)
    scores[ridge] = scoreRidge
    
    '''Decission Tree Regression'''
    decisionTreeRegressor = DecisionTreeRegressor(random_state=0)
    decisionTreeRegressor.fit(X_train, y_train)
    scoreDecisionTreeRegressor = decisionTreeRegressor.score(X_test, y_test)
    scores[decisionTreeRegressor] = scoreDecisionTreeRegressor
    
    import operator
    return max(scores.items(), key=operator.itemgetter(1))[0]

def creacionMatrizDePrediccionDeErroresFuturosRegresion(datosArrayClass, datosInferClass, datosAdicionales, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest, ventana, dia):
    
    '''Sumamos los dias a la ventana para no quedarnos sin matriz al reducirla'''
    ventana = int(ventana) + dia
    
    '''Creamos la matriz completa con todos los dias disponibles en el conjunto de datos'''
    vectorCompleto = vectorCompletoInvertido(datosArrayClass)
    matrizCompleta = crearMatriz(vectorCompleto, ventana)
    
    '''Eliminamos tantas columnas como dias, teniendo cuidado de no eliminar la primera columna que sera el valor a predecir'''
    columnasBorrar = np.arange(dia)
    matrizCompleta = np.delete(matrizCompleta, matrizCompleta[0].size-2-columnasBorrar, 1)
    
    '''Creamos las matrices correspondientes al conjunto de entrenamiento y de pruebas'''
    matrizTrain, matrizTest = matrizEntrenamientoTest(matrizCompleta, datosInferClass, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest)
    
    
    if len(datosAdicionales) != 0:
        for aux in datosAdicionales:
            '''Leemos los datos'''
            ruta = '\predicterapp\static\predicterapp\myDates\dataframe\\' + aux + '.infer';
            datosInferAdicionales = pd.read_pickle(BASE_DIR + ruta)
            datosArrayAdicionales = np.load(BASE_DIR + '\\predicterapp\\static\\predicterapp\\myDates\\narray\\' + aux + '.npy')
            
            '''Creamos el vector y la matriz completa de los datos adicionales'''
            vectorCompletoDatosAdicionales = vectorCompletoInvertido(datosArrayAdicionales)
            matrizCompletaDatosAdicionales = crearMatriz(vectorCompletoDatosAdicionales, ventana)
            
            '''Creamos las matrices correspondientes a los vectores de entrenamiento y a los de pruebas'''
            matrizTrainAdicionales, matrizTestAdicionales = matrizEntrenamientoTest(matrizCompletaDatosAdicionales, datosInferAdicionales, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest)
            
            '''Nos quedamos con la X que es lo unico que nos interesa de las matrices creadas con los datos adicionales'''
            X_trainAdicionales = np.delete(matrizTrainAdicionales, matrizTrainAdicionales[0].size-1, 1)
            X_testAdicionales = np.delete(matrizTestAdicionales, matrizTestAdicionales[0].size-1, 1)
            
            '''Eliminamos tantas columnas como dias, teniendo cuidado de no eliminar la primera columna que sera el valor a predecir'''
            X_trainAdicionales = np.delete(X_trainAdicionales, X_trainAdicionales[0].size-1-columnasBorrar, 1)
            X_testAdicionales = np.delete(X_testAdicionales, X_testAdicionales[0].size-1-columnasBorrar, 1)
            
            '''Unimos le conjunto de datos adicionales'''
            matrizTrain = np.concatenate((X_trainAdicionales, matrizTrain), axis = 1)
            matrizTest = np.concatenate((X_testAdicionales, matrizTest), axis = 1)
    
    return matrizTrain, matrizTest

def calcularScoreMaeDiasAnteriores(matrizTrainN, matrizTestN, clf):
    y_trainN = matrizTrainN[:,matrizTrainN[0].size-1]
    X_trainN = np.delete(matrizTrainN, matrizTrainN[0].size-1, 1)
    y_testN = matrizTestN[:,matrizTestN[0].size-1]
    X_testN = np.delete(matrizTestN, matrizTestN[0].size-1, 1)
    
    clf.fit(X_trainN, y_trainN)
    
    scoreN = clf.score(X_testN, y_testN)
    maeN = mean_absolute_error(y_testN, clf.predict(X_testN))
    
    return scoreN, maeN
    