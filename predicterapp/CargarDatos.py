'''
Created on 22 jul. 2018

@author: Santiago
'''
import pandas_datareader.data as web
import datetime
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datosYahoo = {"BBVA" : "BBVA.MC", "Santander" : "SAN.MC", "Sabadell" : "SAB.MC", "Bankinter" : "BKT.MC", "ACCIONA" : "ANA.MC", "ENDESA" : "ELE.MC", "IBERDROLA" : "IBE.MC", "INDRA" : "IDR.MC", "MAPFRE" : "MAP.MC", "REPSOL" : "REP.MC", "Telefonica" : "TEF.MC"}
datosYahooTupla = (('BBVA', 'BBVA'), ('Santander', 'Santander'), ('Sabadell', 'Sabadell'), ('Bankinter', 'Bankinter'), ('ACCIONA', 'ACCIONA'), ('ENDESA', 'ENDESA'), ('IBERDROLA', 'IBERDROLA'), ('INDRA', 'INDRA'), ('MAPFRE', 'MAPFRE'), ('REPSOL', 'REPSOL'), ('Telefonica', 'Telefonica'), ('DolarVsEuro', 'DolarVsEuro'))

def obtenerDatosApi():
    try:
        for x in datosYahoo:
            ruta = '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer';
            pd.read_pickle(BASE_DIR + ruta)
            hayDatos = True
    except FileNotFoundError:
        hayDatos = False
    
    print (hayDatos)
    
    datoActu = datosActualizado()
    
    if (hayDatos==True) & (datoActu==True):
        print('Todos los datos estan actualizados')
    
    elif (hayDatos==True) & (datoActu==False):
        actualizarDatos()
        
    elif hayDatos==False:
        obtenicionDeDatos()
        
        
def datosActualizado():
    actualizado = True
    current = datetime.datetime.now()
    try:
        for x in datosYahoo:
            ruta = pd.read_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer')
            ultimaFechaCierre = ruta.tail(1).reset_index()['Date'][0]
            if ultimaFechaCierre.date() < current.date():
                actualizado = False
    
    except:
        actualizado = False
        print(actualizado)
    
    return actualizado

def actualizarDatos():
    for x in datosYahoo:
        ruta = pd.read_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer')
        ultimaFechaCierre = ruta.tail(1).reset_index()['Date'][0] + pd.Timedelta(days=1)
        
        peticionApiActualizarDatos(ultimaFechaCierre, x)
   
    
def obtenicionDeDatos():
    '''Seleccionamos las fechas de las que queremos obtener los datos'''
    start = datetime.datetime(2003, 1, 1)
    end = datetime.datetime.now()
    
    for x in datosYahoo:
        peticionApiObtencionDeDatos(start, end, x)
    
def peticionApiActualizarDatos(ultimaFecha, dato):
    current = datetime.datetime.now()
    
    for x in range(0, 3):
        try:
            dataframe = pd.read_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + dato + '.infer')
            
            f = web.DataReader(datosYahoo[dato], 'yahoo', ultimaFecha, current)
            
            dataframeActualizado = f.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
            
            dataframe = dataframe.append(dataframeActualizado)
            
            dataframe = dataframe[~dataframe.index.duplicated()]
            
            dataframe.to_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + dato + '.infer')
            
            break
        except:
            print("Volviendo a intentar la peticion de actualizar datos")

def peticionApiObtencionDeDatos(start, end, dato):
    '''Construimos la peticion a la api de yahoo, intentamos realizar la peticion 3 veces antes de desistir'''
    for x in range(0, 3):
        try:
            f = web.DataReader(datosYahoo[dato], 'yahoo', start, end)
            
            '''Obtenemos los datos y los almacenamos en un dataframe'''
            dataframe = f.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    
            '''Guardamos los datos en un fichero .infer para su posterior procesamiento, pero
            antes comprobamos que los ficheros no existen'''
   
            dataframe.to_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + dato + '.infer')
            break
        except:
            print("Volviendo a intentar la peticion de obtencion de datos")


    
    