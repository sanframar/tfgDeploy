import pandas_datareader.data as web
from sklearn import preprocessing
import datetime
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from predicterapp.CargarDatos import obtenerDatosApi, datosYahoo
from predicterapp.Utils import *

datosYahooConEuro = {"BBVA" : "BBVA.MC","DOLARVSEURO": "DOLARVSEURO", "Santander" : "SAN.MC", "Sabadell" : "SAB.MC", "Bankinter" : "BKT.MC", "ACCIONA" : "ANA.MC", "ENDESA" : "ELE.MC", "IBERDROLA" : "IBE.MC", "INDRA" : "IDR.MC", "MAPFRE" : "MAP.MC", "REPSOL" : "REP.MC", "Telefonica" : "TEF.MC"}

def preProcesamientoDatos():
    try:
        for x in datosYahooConEuro:
            dataframeOld = pd.read_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer')
            dataframe = diasFaltantes(dataframeOld)
            '''Guardamos el nuevo dataframe'''
            dataframe.to_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer')
            
            arrayNulos = indicesNulos(dataframe)
            arraySinNulos = calcularValoresNaN(dataframe.values, arrayNulos, x)
            normalizacionDataframe(arraySinNulos, x)
    except:
        print('Error al realizar el pre procesamiento')
    

#Metodo que devuelve un array con los indices de los valores nulos del dataframe (NaN)
def indicesNulos(dataframe):
    indicesNaN = pd.isnull(dataframe).any(1).nonzero()[0]
    #Modificamos el orden del array para que calcule primero los valores de los extremos
    a = indicesNaN[indicesNaN.size-1:]
    b = indicesNaN[:1]
    c = indicesNaN[1:-1]
    indicesNaN = np.concatenate((b,a,c),axis=0)
    return indicesNaN
    
#Metodo que devuelve un array con los elementos normalizados entre los valores 0 y 1
def normalizacionDataframe(array, dataframeName):
    newArray = preprocessing.normalize(array, axis=0, norm='max')
    guardarArray(newArray, dataframeName)
    
#Metodo que desnormaliza los datos para mostrarlos al usuario
def desNormalizar(valorBaseSinNormalizar, valorBaseNormalizado, datosArray):
    result = []
    for idx, value in enumerate(datosArray):
        if idx == 0:
            diferenciaPorcentaje = diferenciaPorcentual(value, valorBaseNormalizado)
            valor = aplicarPorcentaje(valorBaseSinNormalizar, diferenciaPorcentaje)
            result.append(valor)
        if idx > 0:
            diferenciaPorcentaje = diferenciaPorcentual(value, datosArray[idx-1])
            valor = aplicarPorcentaje(result[len(result)-1], diferenciaPorcentaje)
            result.append(valor)
            
    return result

def guardarArray(array, dataframeName):
    np.save(BASE_DIR + '\\predicterapp\\static\\predicterapp\\myDates\\narray\\' + dataframeName, array)

#Metodo que calcula los nuevos valores para eliminar los NaN de nuestro conjunto de datos
def calcularValoresNaN(array, indicesNaN, dataframeName):
    for aux in indicesNaN:
        if(aux==0):
            nanEnInicio(array, indicesNaN, aux)
        if(aux==array.size-1):
            nanEnFinal(array, indicesNaN, aux)
        if((aux>0) & (aux<array.size-1)):
            nanEnMedio(array, indicesNaN, aux)
    return array
            
def nanEnInicio(array, indicesNaN, aux):
    for idx, value in enumerate(range(0,11)):
        if(np.isnan(array[aux+idx+1]) != True):
            nextValue = array[aux+idx+1]
            nextIndice = aux+idx+1
            break
        
    for idx, value in enumerate(range(0,11)):
        if(np.isnan(array[idx+nextIndice+1]) != True):
            nextValue2 = array[idx+nextIndice+1]
            nextIndice2 = idx+nextIndice+1
            break
    
    newValue = (nextValue + nextValue2)/2
    array[aux] = newValue
    
def nanEnFinal(array, indicesNaN, aux):
    for idx, value in enumerate(range(0,11)):
        if(np.isnan(array[aux-idx-1]) != True):
            nextValue = array[aux-idx-1]
            nextIndice = aux-idx-1
            break
            
    for idx, value in enumerate(range(0,11)):
        if(np.isnan(array[nextIndice-idx-1]) != True):
            nextValue2 = array[nextIndice-idx-1]
            nextIndice2 = nextIndice-idx-1
            break
            
    newValue = (nextValue + nextValue2)/2
    array[aux] = newValue

def nanEnMedio(array, indicesNaN, aux):
    for idx, value in enumerate(range(0,11)):
        if(np.isnan(array[aux+idx+1]) != True):
            nextValue = array[aux+idx+1]
            nextIndice = aux+idx+1
            break
            
    for idx, value in enumerate(range(0,11)):
        if(np.isnan(array[aux-idx-1]) != True):
            afterValue = array[aux-idx-1]
            afterIndice = aux-idx-1
    
    newValue = (nextValue + afterValue)/2
    array[aux] = newValue

def diasFaltantes(dataframe):
    end = datetime.datetime.now().strftime("%Y-%m-%d")
    diasLaborables = pd.bdate_range('2003-01-01', '2018-12-07')
    for aux in diasLaborables:
        find = buscarFecha(aux, dataframe)
        if(find==-1):
            dataframe.loc[aux] = np.NaN
    dataframe = dataframe.sort_index()
    return dataframe

def buscarFecha(fecha, dataframe):
    try:
        valor = dataframe.loc[fecha]
        return 1
    except:
        return -1