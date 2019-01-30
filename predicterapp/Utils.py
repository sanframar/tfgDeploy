

#Calculamos la diferencia porcentual de los dos valores pasados como parametros
def diferenciaPorcentual(valorNuevo, valorAnterior):
    if valorNuevo == valorAnterior:
        return 0
    try:
        return ((valorNuevo - valorAnterior) / valorAnterior) * 100.0
    except ZeroDivisionError:
        return -999999
    
    
#Aplicamos el porcentaje de subida o bajada al valor
def aplicarPorcentaje(valor, porcentaje):
    return valor+(valor*(porcentaje/100))