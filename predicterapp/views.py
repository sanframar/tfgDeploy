from django.shortcuts import render
from  django.http  import  HttpResponse
from django.template import loader
from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect

import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import os
from predicterapp.CargarDatos import obtenerDatosApi, datosYahoo
from predicterapp.PreProcesamiento import preProcesamientoDatos
from predicterapp.Regresion import regresionPolinomial
from .forms import FormularioRegresion
from predicterapp.forms import FormularioClasificacion
from predicterapp.Clasificacion import algoritmoClasificacion
from predicterapp.CargarDatosBCE import obtenerDatosBCE

datosYahooConEuro = {"BBVA" : "BBVA.MC", "Santander" : "SAN.MC", "Sabadell" : "SAB.MC", "Bankinter" : "BKT.MC", "ACCIONA" : "ANA.MC", "ENDESA" : "ELE.MC", "IBERDROLA" : "IBE.MC", "INDRA" : "IDR.MC", "MAPFRE" : "MAP.MC", "REPSOL" : "REP.MC", "Telefonica" : "TEF.MC", "DOLARVSEURO": "DOLARVSEURO"}
# -*- coding: utf-8 -*-
# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def  index ( request ): 
    return render_to_response('predicterapp/index.html')

def datos(request):
    obtenerDatosApi()
    obtenerDatosBCE()
    preProcesamientoDatos()
    BBDD = []
    try:
        for x in datosYahooConEuro:
            datos = pd.read_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer')
            #datos = np.load(BASE_DIR + '\\predicterapp\\static\\predicterapp\\myDates\\narray\\' + x + '.npy')
            fechaInicio = datos.head(1).reset_index()['Date'][0]
            fechaFin = datos.tail(1).reset_index()['Date'][0]
            tamDatos = datos.size
            
            #fechaInicio = '2003-01-01'
            #fechaFin = '2018-12-07'
            #tamDatos = datos.size
            
            tupla = [x, fechaInicio, fechaFin, tamDatos]
            BBDD.append(tupla)
    except:
        fechaInicio = "Sin fecha inicio"
        fechaFin = "Sin fecha fin"
        tamDatos = 0
    
    template = loader.get_template('predicterapp/datos.html')
    context = {
        'BBDD': BBDD,
    }
    return HttpResponse(template.render(context, request))

def preProcesamiento(request):
    #preProcesamientoDatos()
    BBDD = []
    try:
        for x in datosYahooConEuro:
            datosArray = np.load(BASE_DIR + '\\predicterapp\\static\\predicterapp\\myDates\\narray\\' + x + '.npy')
            datosDataframe = pd.read_pickle(BASE_DIR + '\predicterapp\static\predicterapp\myDates\dataframe\\' + x + '.infer')
        
            #Estadisticas del conjunto de datos pre procesados
            valorMinimoArray = datosArray.min()
            valorMaximoArray = datosArray.max()
            valorMedioArray = np.mean(datosArray)
            valoresNaNArray = np.isnan(datosArray).sum()
            
            #Estadisticas del conjunto de datos sin el pre procesado
            valorMinimoDataframe = datosDataframe.loc[datosDataframe['Close'].idxmin()][0]
            valorMaximoDataframe = datosDataframe.loc[datosDataframe['Close'].idxmax()][0]
            valorMedioDataframe = datosDataframe['Close'].median()
            valoresNaNDataframe = datosDataframe['Close'].isnull().sum()  + (datosArray.size - datosDataframe.size)
            
            tupla = [x, valoresNaNArray, valorMinimoArray, valorMaximoArray, valorMedioArray, valorMinimoDataframe, valorMaximoDataframe, valorMedioDataframe, valoresNaNDataframe]
            BBDD.append(tupla)
    except:
        print("Error al mostrar los datos del pre-procesamiento")
    
    template = loader.get_template('predicterapp/preProcesamiento.html')
    context = {
        'BBDD': BBDD,
    }
    return HttpResponse(template.render(context, request))

def regresion ( request ): 
    datosArray = []
    form = FormularioRegresion()
    for aux in datosYahoo:
        datosArray.append(aux)
    template = loader.get_template('predicterapp/regresion.html')
    context = {
        'datosArray': datosArray,
        'form': form,
    }
    return HttpResponse(template.render(context, request))

def formularioParaRegresion(request):
    # if this is a POST request we need to process the form data
    if request.method == 'GET':
        # create a form instance and populate it with data from the request:
        form = FormularioRegresion(request.GET)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # redirect to a new URL:
            #return HttpResponseRedirect(resultadoRegresion(form))
            selectMulti = request.GET.getlist('selectMulti')
            return resultadoRegresion(form, selectMulti)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = FormularioRegresion()
        return render(request, 'predicterapp/regresion.html', {'form': form})
    return render(request, 'predicterapp/regresion.html', {'form': form})

def resultadoRegresion(form, selectMulti):
    ventana = form.data['ventana']
    diasAPredecir = form.data['diasAPredecir']
    select = form.data['select']
    fechaInicioTrain = form.data['fechaIniTrain']
    fechaFinTrain = form.data['fechaFinTrain']
    fechaInicioTest = form.data['fechaIniTest']
    fechaFinTest = form.data['fechaFinTest']
            
    score, mae, prediccion = regresionPolinomial(select, selectMulti, ventana, diasAPredecir, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest)
    
    template = loader.get_template('predicterapp/resultadoRegresion.html')
    myList = zip(score, mae, prediccion)
    context = {
        'ventana': ventana,
        'diasAPredecir': diasAPredecir,
        'select': select,
        'myList': myList,
    }
    return HttpResponse(template.render(context))


def clasificacion ( request ): 
    datosArray = []
    form = FormularioClasificacion()
    for aux in datosYahoo:
        datosArray.append(aux)
    template = loader.get_template('predicterapp/clasificacion.html')
    context = {
        'datosArray': datosArray,
        'form': form,
    }
    return HttpResponse(template.render(context, request))

def formularioParaClasificacion(request):
    # if this is a POST request we need to process the form data
    if request.method == 'GET':
        # create a form instance and populate it with data from the request:
        form = FormularioClasificacion(request.GET)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # redirect to a new URL:
            #return HttpResponseRedirect(resultadoRegresion(form))
            selectMulti = request.GET.getlist('selectMulti')
            return resultadoClasificacion(form, selectMulti)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = FormularioClasificacion()
        return render(request, 'predicterapp/clasificacion.html', {'form': form})
    return render(request, 'predicterapp/clasificacion.html', {'form': form})

def resultadoClasificacion(form, selectMulti):
    ventana = form.data['ventana']
    diasAPredecir = form.data['diasAPredecir']
    epsilon = form.data['epsilon']
    select = form.data['select']
    fechaInicioTrain = form.data['fechaIniTrain']
    fechaFinTrain = form.data['fechaFinTrain']
    fechaInicioTest = form.data['fechaIniTest']
    fechaFinTest = form.data['fechaFinTest']
            
    score, prediccion = algoritmoClasificacion(select, selectMulti, ventana, diasAPredecir, fechaInicioTrain, fechaFinTrain, fechaInicioTest, fechaFinTest, epsilon)
    
    template = loader.get_template('predicterapp/resultadoClasificacion.html')
    myList = zip(score, prediccion)
    context = {
        'ventana': ventana,
        'diasAPredecir': diasAPredecir,
        'select': select,
        'myList': myList,
    }
    return HttpResponse(template.render(context))

def supervisado(request):
    return render_to_response('predicterapp/supervisado.html')

def noSupervisado(request):
    return render_to_response('predicterapp/noSupervisado.html')