from django.test import TestCase

from predicterapp.Utils import *

from django.test import TestCase

class UtilsTests(TestCase):
    
    def test_diferencia_porcentual_igual(self):
            """
            Comprobamos que al ser los dos valores iguales el sistema responde
            adecuadamente con un cero
            """
            valorNuevo = 10
            valorAnterior = 10
            self.assertEqual(diferenciaPorcentual(valorNuevo, valorAnterior), 0)

    def test_diferencia_porcentual_menor(self):
            """
            Comprobamos que al dividir entre cero el sistema muestra
            como resultado -999999
            """
            valorNuevo = 10
            valorAnterior = 0
            self.assertEqual(diferenciaPorcentual(valorNuevo, valorAnterior), -999999)