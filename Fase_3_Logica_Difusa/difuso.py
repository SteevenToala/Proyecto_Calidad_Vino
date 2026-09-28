# Proyecto Inteligencia Artificial
# Fase 3: Sistema de Inferencia Difusa Mamdani Continuo
# Superacion del problema de fronteras rigidas (Boundary Problem)

import numpy as np

# -----------------------------------------------------------------------------
# Seccion 1: Funciones de Pertenencia (Triangulares y Trapezoidales)
# -----------------------------------------------------------------------------
def pertenencia_baja(valor, corte_bajo, punto_medio):
    if valor <= corte_bajo:
        return 1.0
    elif valor >= punto_medio:
        return 0.0
    else:
        denominador = punto_medio - corte_bajo
        return (punto_medio - valor) / denominador if denominador > 0 else 0.0

def pertenencia_media(valor, corte_bajo, punto_medio, corte_alto):
    if valor <= corte_bajo or valor >= corte_alto:
        return 0.0
    elif valor == punto_medio:
        return 1.0
    elif valor < punto_medio:
        denominador = punto_medio - corte_bajo
        return (valor - corte_bajo) / denominador if denominador > 0 else 0.0
    else:
        denominador = corte_alto - punto_medio
        return (corte_alto - valor) / denominador if denominador > 0 else 0.0

def pertenencia_alta(valor, punto_medio, corte_alto):
    if valor >= corte_alto:
        return 1.0
    elif valor <= punto_medio:
        return 0.0
    else:
        denominador = corte_alto - punto_medio
        return (valor - punto_medio) / denominador if denominador > 0 else 0.0

def pertenencias_variable(valor, cortes_variable):
    return {
        'bajo': pertenencia_baja(valor, cortes_variable['corte_bajo'], cortes_variable['punto_medio']),
        'medio': pertenencia_media(valor, cortes_variable['corte_bajo'], cortes_variable['punto_medio'], cortes_variable['corte_alto']),
        'alto': pertenencia_alta(valor, cortes_variable['punto_medio'], cortes_variable['corte_alto'])
    }

# -----------------------------------------------------------------------------
# Seccion 2: Fuzzificacion (Conversion de Valores Nitidos a Grados Difusos)
# -----------------------------------------------------------------------------
def fuzzificar_perfil_vino(valores_vino, cortes_variables):
    # Transforma los valores quimicos nitidos (crisp) en grados de verdad mu en [0, 1]
    grados_fuzzificados = {}
    for variable, cortes in cortes_variables.items():
        if variable in valores_vino:
            valor_quimico = float(valores_vino[variable])
            grados_fuzzificados[variable] = pertenencias_variable(valor_quimico, cortes)
    return grados_fuzzificados

# -----------------------------------------------------------------------------
# Seccion 3: Inferencia Difusa (Mamdani - Operador T-Norma MIN)
# -----------------------------------------------------------------------------
def evaluar_inferencia_mamdani(grados_fuzzificados, lista_reglas, pesos_reglas=None):
    # Evalua la fuerza de activacion de cada regla aplicando el operador MIN entre antecedentes
    if pesos_reglas is None:
        pesos_reglas = [1.0] * len(lista_reglas)
        
    reglas_disparadas = []
    for indice, regla in enumerate(lista_reglas):
        peso_regla = pesos_reglas[indice] if indice < len(pesos_reglas) else 1.0
        if peso_regla < 0.05:
            continue
            
        grados_antecedentes = []
        for variable, etiqueta in regla['condiciones'].items():
            if variable in grados_fuzzificados:
                grado = grados_fuzzificados[variable].get(etiqueta, 0.0)
                grados_antecedentes.append(grado)
            else:
                grados_antecedentes.append(0.0)
                
        # Operador de interseccion difusa (T-norma MIN de Mamdani)
        fuerza_disparo = min(grados_antecedentes) if grados_antecedentes else 0.0
        disparo_ponderado = fuerza_disparo * peso_regla
        
        if disparo_ponderado > 0.05:
            reglas_disparadas.append({
                'id': regla['id'],
                'consecuente': regla['consecuente'],
                'fuerza_disparo': round(float(fuerza_disparo), 3),
                'peso': round(float(peso_regla), 2),
                'disparo_ponderado': round(float(disparo_ponderado), 3),
                'condiciones': regla['condiciones']
            })
            
    return reglas_disparadas

# -----------------------------------------------------------------------------
# Seccion 4: Agregacion Difusa (Combinacion de Reglas por Clase)
# -----------------------------------------------------------------------------
def agregar_activaciones_clase(reglas_disparadas):
    # Agrega la evidencia de todas las reglas activas acumulando por cada clase consecuente
    puntuaciones_clase = {'ALTA': 0.0, 'MEDIA': 0.0, 'BAJA': 0.0}
    for regla in reglas_disparadas:
        clase_consecuente = regla['consecuente']
        if clase_consecuente in puntuaciones_clase:
            puntuaciones_clase[clase_consecuente] += regla['disparo_ponderado']
            
    total_puntuaciones = sum(puntuaciones_clase.values())
    if total_puntuaciones > 0:
        probabilidades_clase = {
            clase: round(puntos / total_puntuaciones, 3)
            for clase, puntos in puntuaciones_clase.items()
        }
    else:
        probabilidades_clase = {'ALTA': 0.0, 'MEDIA': 1.0, 'BAJA': 0.0}
        
    return puntuaciones_clase, probabilidades_clase

# -----------------------------------------------------------------------------
# Seccion 5: Defuzzificacion (Centroide CoG y Veredicto Final)
# -----------------------------------------------------------------------------
def defuzzificar_veredicto(puntuaciones_clase, probabilidades_clase):
    # Veredicto categorico segun la maxima pertenencia acumulada
    if sum(puntuaciones_clase.values()) > 0:
        clase_predicha = max(puntuaciones_clase, key=puntuaciones_clase.get)
    else:
        clase_predicha = 'MEDIA'
    return clase_predicha

def defuzzificar_centroide_gravedad(probabilidades_clase, puntos_muestreo=400):
    # Defuzzificacion continua mediante el metodo del Centroide de Gravedad (CoG)
    # CoG = sum(z * mu(z)) / sum(mu(z))
    escala_z = np.linspace(0.0, 10.0, puntos_muestreo)
    
    pertenencia_baja_salida = np.clip(np.maximum(0, (5.0 - escala_z) / 2.5), 0, 1)
    pertenencia_baja_salida[escala_z <= 2.5] = 1.0
    pertenencia_media_salida = np.clip(np.minimum((escala_z - 3.5) / 2.5, (8.5 - escala_z) / 2.5), 0, 1)
    pertenencia_alta_salida = np.clip(np.maximum(0, (escala_z - 6.0) / 2.5), 0, 1)
    pertenencia_alta_salida[escala_z >= 8.5] = 1.0
    
    # Recorte Mamdani por operador MIN
    area_baja = np.minimum(pertenencia_baja_salida, probabilidades_clase['BAJA'])
    area_media = np.minimum(pertenencia_media_salida, probabilidades_clase['MEDIA'])
    area_alta = np.minimum(pertenencia_alta_salida, probabilidades_clase['ALTA'])
    area_agregada = np.maximum(area_baja, np.maximum(area_media, area_alta))
    
    denominador_area = np.sum(area_agregada)
    if denominador_area > 0:
        centroide = np.sum(escala_z * area_agregada) / denominador_area
    else:
        centroide = 5.8
    return round(float(centroide), 2)

# -----------------------------------------------------------------------------
# Seccion 6: Funciones de Calculo y Evaluacion en Dataset
# -----------------------------------------------------------------------------
def evaluar_vino_difuso(valores_vino, cortes_variables, lista_reglas, pesos_reglas=None):
    # 1. Fuzzificacion
    grados_fuzzificados = fuzzificar_perfil_vino(valores_vino, cortes_variables)
    
    # 2. Inferencia Difusa (Mamdani T-Norma MIN)
    reglas_disparadas = evaluar_inferencia_mamdani(grados_fuzzificados, lista_reglas, pesos_reglas)
    
    # 3. Agregacion
    puntuaciones_clase, probabilidades_clase = agregar_activaciones_clase(reglas_disparadas)
    
    # 4. Defuzzificacion
    clase_predicha = defuzzificar_veredicto(puntuaciones_clase, probabilidades_clase)
    
    return clase_predicha, probabilidades_clase, reglas_disparadas

def calcular_exactitud_dataset(datos_evaluacion, cortes_variables, lista_reglas, pesos_reglas=None):
    aciertos = 0
    total_filas = len(datos_evaluacion)
    
    for _, fila in datos_evaluacion.iterrows():
        clase_predicha, _, _ = evaluar_vino_difuso(fila, cortes_variables, lista_reglas, pesos_reglas)
        if clase_predicha == fila['calidad']:
            aciertos += 1
            
    exactitud = (aciertos / total_filas) * 100.0
    return round(float(exactitud), 2)

# -----------------------------------------------------------------------------
# Ejecucion directa de la Fase 3
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    import sys
    import matplotlib.pyplot as plt
    
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from Fase_1_Datos.preparacion import cargar_y_preparar
    from Fase_2_PRISM.prism import inducir_reglas_prism
    
    ruta_archivo = "winequality-red.csv"
    if not os.path.exists(ruta_archivo):
        ruta_archivo = os.path.join(os.path.dirname(__file__), "..", "winequality-red.csv")
        
    print("\nFase 3: Inferencia Difusa Mamdani Continua")
    datos_preparados = cargar_y_preparar(ruta_archivo)
    reglas_prism = inducir_reglas_prism(datos_preparados)
    
    exactitud_entrenamiento = calcular_exactitud_dataset(datos_preparados['datos_entrenamiento'], datos_preparados['cortes_base'], reglas_prism)
    exactitud_prueba = calcular_exactitud_dataset(datos_preparados['datos_prueba'], datos_preparados['cortes_base'], reglas_prism)
    print(f"Exactitud Difusa Mamdani en Entrenamiento : {exactitud_entrenamiento}%")
    print(f"Exactitud Difusa Mamdani en Prueba (Test)  : {exactitud_prueba}%")
    
    # Graficar curvas de pertenencia difusa para alcohol
    cortes_alcohol = datos_preparados['cortes_base']['alcohol']
    valores_eje_x = np.linspace(cortes_alcohol['minimo'], cortes_alcohol['maximo'], 300)
    pertenencias_baja_x = [pertenencia_baja(x, cortes_alcohol['corte_bajo'], cortes_alcohol['punto_medio']) for x in valores_eje_x]
    pertenencias_media_x = [pertenencia_media(x, cortes_alcohol['corte_bajo'], cortes_alcohol['punto_medio'], cortes_alcohol['corte_alto']) for x in valores_eje_x]
    pertenencias_alta_x = [pertenencia_alta(x, cortes_alcohol['punto_medio'], cortes_alcohol['corte_alto']) for x in valores_eje_x]
    
    plt.figure(figsize=(7, 4))
    plt.plot(valores_eje_x, pertenencias_baja_x, label='Bajo', color='#2980b9', linewidth=2)
    plt.plot(valores_eje_x, pertenencias_media_x, label='Medio', color='#f39c12', linewidth=2)
    plt.plot(valores_eje_x, pertenencias_alta_x, label='Alto', color='#27ae60', linewidth=2)
    plt.title("Funciones de Pertenencia Difusa (Alcohol)")
    plt.xlabel("Graduacion Alcoholica (% vol.)")
    plt.ylabel("Grado de Pertenencia (mu)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    print("\nMostrando grafico de pertenencias difusas...")
    plt.show()
