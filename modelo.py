# Proyecto Inteligencia Artificial
# Orquestador general del flujo de 4 fases en memoria

import os
import sys

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
sys.path.append(carpeta_actual)

from Fase_1_Datos.preparacion import cargar_y_preparar, MAPA_VARIABLES, JUSTIFICACIONES_QUIMICAS
from Fase_2_PRISM.prism import recubrimiento_secuencial_prism, evaluar_prism_rigido, inducir_reglas_prism
from Fase_3_Logica_Difusa.difuso import (
    evaluar_vino_difuso, calcular_exactitud_dataset,
    pertenencia_baja, pertenencia_media, pertenencia_alta, pertenencias_variable,
    fuzzificar_perfil_vino, evaluar_inferencia_mamdani, agregar_activaciones_clase,
    defuzzificar_veredicto, defuzzificar_centroide_gravedad
)
from Fase_4_Algoritmo_Genetico.genetico import optimizar_con_algoritmo_genetico

# Ejecucion secuencial de las 4 fases de aprendizaje
def entrenar_modelo_completo(ruta_archivo_csv):
    # Fase 1: Carga, analisis de correlaciones y particion 80/20
    datos_preparados = cargar_y_preparar(ruta_archivo_csv)
    
    # Fase 2: Induccion de reglas modulares con PRISM
    reglas_modulares = recubrimiento_secuencial_prism(datos_preparados)
    exactitud_entrenamiento_rigido = evaluar_prism_rigido(datos_preparados['datos_entrenamiento'], reglas_modulares)
    exactitud_prueba_rigido = evaluar_prism_rigido(datos_preparados['datos_prueba'], reglas_modulares)
    
    # Fase 3: Evaluacion con Logica Difusa continua de Mamdani
    exactitud_entrenamiento_inicial = calcular_exactitud_dataset(
        datos_preparados['datos_entrenamiento'], datos_preparados['cortes_base'], reglas_modulares
    )
    exactitud_prueba_inicial = calcular_exactitud_dataset(
        datos_preparados['datos_prueba'], datos_preparados['cortes_base'], reglas_modulares
    )
    
    # Fase 4: Calibracion de cortes difusos y pesos con Algoritmo Genetico
    cortes_optimizados, pesos_optimizados, historial_aptitud, configuracion_genetico = optimizar_con_algoritmo_genetico(
        datos_preparados, reglas_modulares
    )
    
    # Evaluacion final del modelo calibrado
    exactitud_entrenamiento_optimizada = calcular_exactitud_dataset(
        datos_preparados['datos_entrenamiento'], cortes_optimizados, reglas_modulares, pesos_optimizados
    )
    exactitud_prueba_optimizada = calcular_exactitud_dataset(
        datos_preparados['datos_prueba'], cortes_optimizados, reglas_modulares, pesos_optimizados
    )
    
    cantidad_reglas_activas = sum(1 for peso in pesos_optimizados if peso > 0.1)
    
    return {
        'datos': datos_preparados,
        'reglas': reglas_modulares,
        'cortes_base': datos_preparados['cortes_base'],
        'cortes_optimizados': cortes_optimizados,
        'pesos_optimizados': pesos_optimizados,
        'exactitud_entrenamiento_rigido': exactitud_entrenamiento_rigido,
        'exactitud_prueba_rigido': exactitud_prueba_rigido,
        'exactitud_entrenamiento_inicial': exactitud_entrenamiento_inicial,
        'exactitud_entrenamiento_optimizada': exactitud_entrenamiento_optimizada,
        'exactitud_prueba_inicial': exactitud_prueba_inicial,
        'exactitud_prueba_optimizada': exactitud_prueba_optimizada,
        'reglas_activas': cantidad_reglas_activas,
        'total_reglas': len(reglas_modulares),
        'historial_aptitud': historial_aptitud,
        'configuracion_genetico': configuracion_genetico,
        # Claves de compatibilidad
        'acc_train_rigido': exactitud_entrenamiento_rigido,
        'acc_test_rigido': exactitud_prueba_rigido,
        'acc_train_inicial': exactitud_entrenamiento_inicial,
        'acc_train_optimizada': exactitud_entrenamiento_optimizada,
        'acc_test_inicial': exactitud_prueba_inicial,
        'acc_test_optimizada': exactitud_prueba_optimizada,
        'historial_fitness': historial_aptitud,
        'config_ag': configuracion_genetico
    }
