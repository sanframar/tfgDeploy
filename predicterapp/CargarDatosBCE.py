import pandas_datareader.data as web
import datetime
import pandas as pd
import requests
from xml.etree import ElementTree
import xml.etree.ElementTree as ET

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

anoInicio = '2003'
annoFin = str(datetime.datetime.now().year)

def obtenerDatosBCE():
    '''Realizamos la peticion a la API del BCE'''
    data = requests.get('https://sdw-wsrest.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=' + anoInicio + '&endPeriod=' + annoFin)
    
    '''Obtenemos el arbol xml que nos da como respuesta a nuestra peticion'''
    root = ET.fromstring(data.content)
    
    '''Recorremos el arbol y vamos sacando los valores que necesitamos'''
    diccionario = {}
    for child in root.findall('{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message}DataSet'):
        for child1 in child.findall('{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Series'):
            for child2 in child1.findall('{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Obs'):
                fecha = child2.find('{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsDimension')
                valor = child2.find('{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}ObsValue')
                #print(child2.tag, child2.attrib)
                #print(fecha.attrib['value'], valor.attrib['value'])
                fechaValor = datetime.datetime.strptime(fecha.attrib['value'], '%Y-%m-%d')
                valorValor = float(valor.attrib['value'])
                diccionario[fechaValor] = valorValor
    
    '''Creamos nuestro dataframe con los datos obtenidos'''
    listaFecha = diccionario.keys()
    listaValor = diccionario.values()
    d = {'Close': list(listaValor)}
    df = pd.DataFrame(data=d, index=listaFecha)
    df.index.name='Date'
    
    '''Guardamos nuestro dataframe en un fichero infer'''
    df.to_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + 'DOLARVSEURO' + '.infer')