# Proyecto Inteligencia Artificial
# Calibracion de hiperparametros difusos mediante Algoritmo Genetico
# Basado en la teoria de John H. Holland (1975)

import copy
import random
import numpy as np

# -----------------------------------------------------------------------------
# Seccion 1: Hiperparametros de Optimizacion y Probabilidades
# -----------------------------------------------------------------------------
TAMANO_POBLACION = 25
NUMERO_GENERACIONES = 25
PROBABILIDAD_CRUCE = 0.85
PROBABILIDAD_MUTACION = 0.20
METODO_SELECCION = 'torneo'
TAMANO_TORNEO = 3
CANTIDAD_ELITE = 2

# -----------------------------------------------------------------------------
# Seccion 2: Poblacion Inicial y Representacion Cromosomica
# -----------------------------------------------------------------------------
def inicializar_poblacion(cortes_base, cantidad_reglas, tamano_poblacion=TAMANO_POBLACION):
    individuo_base = {
        'cortes': copy.deepcopy(cortes_base),
        'pesos': [1.0] * cantidad_reglas
    }
    
    poblacion = [individuo_base]
    for _ in range(tamano_poblacion - 1):
        nuevo_individuo = copy.deepcopy(individuo_base)
        for variable, limites in nuevo_individuo['cortes'].items():
            rango_variable = max(1e-4, limites['maximo'] - limites['minimo'])
            variacion_aleatoria = (random.random() - 0.5) * 0.08 * rango_variable
            
            limites['corte_bajo'] = round(float(np.clip(
                limites['corte_bajo'] + variacion_aleatoria,
                limites['minimo'],
                limites['punto_medio'] - 0.01 * rango_variable
            )), 4)
            limites['punto_medio'] = round(float(np.clip(
                limites['punto_medio'] + variacion_aleatoria * 0.7,
                limites['corte_bajo'] + 0.01 * rango_variable,
                limites['corte_alto'] - 0.01 * rango_variable
            )), 4)
            limites['corte_alto'] = round(float(np.clip(
                limites['corte_alto'] + variacion_aleatoria * 0.5,
                limites['punto_medio'] + 0.01 * rango_variable,
                limites['maximo']
            )), 4)
            
        for indice_regla in range(cantidad_reglas):
            nuevo_individuo['pesos'][indice_regla] = round(float(random.uniform(0.6, 1.4)), 3)
            
        poblacion.append(nuevo_individuo)
        
    return poblacion

# -----------------------------------------------------------------------------
# Seccion 3: Evaluacion de la Funcion de Aptitud (Fitness)
# -----------------------------------------------------------------------------
def calcular_aptitud_individuo(individuo, columnas_datos, nombres_variables, reglas_prism, calidades_esperadas, total_muestras):
    cortes = individuo['cortes']
    pesos = individuo['pesos']
    pertenencias_variables = {}
    
    for variable in nombres_variables:
        valores = columnas_datos[variable]
        corte_bajo = cortes[variable]['corte_bajo']
        punto_medio = cortes[variable]['punto_medio']
        corte_alto = cortes[variable]['corte_alto']
        
        denominador_bajo = max(1e-5, punto_medio - corte_bajo)
        pertenencia_baja = np.clip((punto_medio - valores) / denominador_bajo, 0.0, 1.0)
        
        denominador_subida = max(1e-5, punto_medio - corte_bajo)
        denominador_bajada = max(1e-5, corte_alto - punto_medio)
        tramo_subida = (valores - corte_bajo) / denominador_subida
        tramo_bajada = (corte_alto - valores) / denominador_bajada
        pertenencia_media = np.clip(np.minimum(tramo_subida, tramo_bajada), 0.0, 1.0)
        
        denominador_alto = max(1e-5, corte_alto - punto_medio)
        pertenencia_alta = np.clip((valores - punto_medio) / denominador_alto, 0.0, 1.0)
        
        pertenencias_variables[variable] = {
            'bajo': pertenencia_baja,
            'medio': pertenencia_media,
            'alto': pertenencia_alta
        }
        
    puntuacion_alta = np.zeros(total_muestras)
    puntuacion_media = np.zeros(total_muestras)
    puntuacion_baja = np.zeros(total_muestras)
    
    for indice, regla in enumerate(reglas_prism):
        peso_regla = pesos[indice]
        if peso_regla < 0.05:
            continue
        activacion_regla = None
        for variable, etiqueta in regla['condiciones'].items():
            if variable in pertenencias_variables:
                grado = pertenencias_variables[variable][etiqueta]
                activacion_regla = grado if activacion_regla is None else np.minimum(activacion_regla, grado)
        if activacion_regla is not None:
            activacion_ponderada = activacion_regla * peso_regla
            consecuente = regla['consecuente']
            if consecuente == 'ALTA':
                puntuacion_alta += activacion_ponderada
            elif consecuente == 'MEDIA':
                puntuacion_media += activacion_ponderada
            elif consecuente == 'BAJA':
                puntuacion_baja += activacion_ponderada
                
    matriz_puntuaciones = np.vstack([puntuacion_alta, puntuacion_media, puntuacion_baja])
    indices_prediccion = np.argmax(matriz_puntuaciones, axis=0)
    indices_prediccion[np.sum(matriz_puntuaciones, axis=0) < 1e-4] = 1 # MEDIA por defecto
    
    etiquetas_calidad = np.array(['ALTA', 'MEDIA', 'BAJA'])
    predicciones_finales = etiquetas_calidad[indices_prediccion]
    exactitud = (np.sum(predicciones_finales == calidades_esperadas) / total_muestras) * 100.0
    return round(float(exactitud), 2)

# -----------------------------------------------------------------------------
# Seccion 4: Metodos de Seleccion de Padres (Ruleta y Torneo)
# -----------------------------------------------------------------------------
def seleccion_ruleta(poblacion, aptitudes):
    suma_aptitudes = sum(aptitudes)
    if suma_aptitudes <= 0:
        return random.choice(poblacion)
        
    probabilidades_seleccion = [aptitud / suma_aptitudes for aptitud in aptitudes]
    probabilidades_acumuladas = np.cumsum(probabilidades_seleccion)
    tiro_ruleta = random.random()
    
    for indice, limite_acumulado in enumerate(probabilidades_acumuladas):
        if tiro_ruleta <= limite_acumulado:
            return poblacion[indice]
            
    return poblacion[-1]

def seleccion_torneo(poblacion, aptitudes, tamano_torneo=TAMANO_TORNEO):
    indices_participantes = random.sample(range(len(poblacion)), tamano_torneo)
    indice_ganador = max(indices_participantes, key=lambda indice: aptitudes[indice])
    return poblacion[indice_ganador]

def seleccionar_padre(poblacion, aptitudes, metodo_seleccion=METODO_SELECCION, tamano_torneo=TAMANO_TORNEO):
    if metodo_seleccion == 'ruleta':
        return seleccion_ruleta(poblacion, aptitudes)
    return seleccion_torneo(poblacion, aptitudes, tamano_torneo)

# -----------------------------------------------------------------------------
# Seccion 5: Cruzamiento con Probabilidad de Cruce (Pc)
# -----------------------------------------------------------------------------
def cruzamiento(primer_padre, segundo_padre, nombres_variables, cantidad_reglas, probabilidad_cruce=PROBABILIDAD_CRUCE):
    valor_aleatorio = random.random()
    
    if valor_aleatorio < probabilidad_cruce:
        primer_hijo = {'cortes': copy.deepcopy(primer_padre['cortes']), 'pesos': list(primer_padre['pesos'])}
        segundo_hijo = {'cortes': copy.deepcopy(segundo_padre['cortes']), 'pesos': list(segundo_padre['pesos'])}
        
        for variable in nombres_variables:
            if random.random() < 0.5:
                primer_hijo['cortes'][variable] = copy.deepcopy(segundo_padre['cortes'][variable])
                segundo_hijo['cortes'][variable] = copy.deepcopy(primer_padre['cortes'][variable])
                
        for indice in range(cantidad_reglas):
            if random.random() < 0.5:
                primer_hijo['pesos'][indice] = segundo_padre['pesos'][indice]
                segundo_hijo['pesos'][indice] = primer_padre['pesos'][indice]
                
        return primer_hijo, segundo_hijo
    else:
        return copy.deepcopy(primer_padre), copy.deepcopy(segundo_padre)

# -----------------------------------------------------------------------------
# Seccion 6: Mutacion con Probabilidad de Mutacion (Pm)
# -----------------------------------------------------------------------------
def mutacion(individuo, nombres_variables, cortes_base, cantidad_reglas, probabilidad_mutacion=PROBABILIDAD_MUTACION):
    valor_aleatorio = random.random()
    
    if valor_aleatorio < probabilidad_mutacion:
        individuo_mutado = copy.deepcopy(individuo)
        
        for variable in nombres_variables:
            if random.random() < 0.35:
                limite_minimo = cortes_base[variable]['minimo']
                limite_maximo = cortes_base[variable]['maximo']
                rango_variable = max(1e-4, limite_maximo - limite_minimo)
                delta_ajuste = (random.random() - 0.5) * 0.07 * rango_variable
                
                corte_bajo = individuo_mutado['cortes'][variable]['corte_bajo'] + delta_ajuste
                punto_medio = individuo_mutado['cortes'][variable]['punto_medio'] + delta_ajuste * 0.8
                corte_alto = individuo_mutado['cortes'][variable]['corte_alto'] + delta_ajuste * 0.6
                
                corte_bajo = max(limite_minimo, min(corte_bajo, limite_maximo - 0.03 * rango_variable))
                punto_medio = max(corte_bajo + 0.01 * rango_variable, min(punto_medio, limite_maximo - 0.02 * rango_variable))
                corte_alto = max(punto_medio + 0.01 * rango_variable, min(corte_alto, limite_maximo))
                
                individuo_mutado['cortes'][variable]['corte_bajo'] = round(float(corte_bajo), 4)
                individuo_mutado['cortes'][variable]['punto_medio'] = round(float(punto_medio), 4)
                individuo_mutado['cortes'][variable]['corte_alto'] = round(float(corte_alto), 4)
                
        for indice in range(cantidad_reglas):
            if random.random() < 0.25:
                ajuste_peso = (random.random() - 0.5) * 0.35
                peso_actualizado = max(0.0, min(1.8, individuo_mutado['pesos'][indice] + ajuste_peso))
                individuo_mutado['pesos'][indice] = round(float(peso_actualizado), 3)
                
        return individuo_mutado
    else:
        return individuo

# -----------------------------------------------------------------------------
# Seccion 7: Ciclo Evolutivo, Elitismo y Criterio de Parada
# -----------------------------------------------------------------------------
def optimizar_con_algoritmo_genetico(
    datos_vino, 
    reglas_prism, 
    tamano_poblacion=TAMANO_POBLACION, 
    numero_generaciones=NUMERO_GENERACIONES,
    probabilidad_cruce=PROBABILIDAD_CRUCE,
    probabilidad_mutacion=PROBABILIDAD_MUTACION,
    metodo_seleccion=METODO_SELECCION,
    cantidad_elite=CANTIDAD_ELITE,
    tamano_torneo=TAMANO_TORNEO
):
    cortes_base = datos_vino['cortes_base']
    datos_entrenamiento = datos_vino['df_train']
    nombres_variables = datos_vino['cols_quimicas']
    calidades_esperadas = datos_entrenamiento['calidad'].values
    total_muestras = len(datos_entrenamiento)
    columnas_datos = {columna: datos_entrenamiento[columna].values for columna in nombres_variables}
    cantidad_reglas = len(reglas_prism)
    
    random.seed(42)
    np.random.seed(42)
    
    poblacion = inicializar_poblacion(cortes_base, cantidad_reglas, tamano_poblacion)
    aptitudes = [
        calcular_aptitud_individuo(
            individuo, columnas_datos, nombres_variables, reglas_prism, calidades_esperadas, total_muestras
        )
        for individuo in poblacion
    ]
    
    indice_mejor = int(np.argmax(aptitudes))
    mejor_individuo = copy.deepcopy(poblacion[indice_mejor])
    mejor_aptitud = aptitudes[indice_mejor]
    historial_aptitud = [mejor_aptitud]
    
    for numero_generacion in range(1, numero_generaciones + 1):
        indices_ordenados = np.argsort(aptitudes)[::-1]
        nueva_poblacion = [copy.deepcopy(poblacion[indice]) for indice in indices_ordenados[:cantidad_elite]]
        
        while len(nueva_poblacion) < tamano_poblacion:
            primer_padre = seleccionar_padre(poblacion, aptitudes, metodo_seleccion, tamano_torneo)
            segundo_padre = seleccionar_padre(poblacion, aptitudes, metodo_seleccion, tamano_torneo)
            
            primer_hijo, segundo_hijo = cruzamiento(
                primer_padre, segundo_padre, nombres_variables, cantidad_reglas, probabilidad_cruce
            )
            
            primer_hijo = mutacion(
                primer_hijo, nombres_variables, cortes_base, cantidad_reglas, probabilidad_mutacion
            )
            nueva_poblacion.append(primer_hijo)
            
            if len(nueva_poblacion) < tamano_poblacion:
                segundo_hijo = mutacion(
                    segundo_hijo, nombres_variables, cortes_base, cantidad_reglas, probabilidad_mutacion
                )
                nueva_poblacion.append(segundo_hijo)
                
        poblacion = nueva_poblacion
        aptitudes = [
            calcular_aptitud_individuo(
                individuo, columnas_datos, nombres_variables, reglas_prism, calidades_esperadas, total_muestras
            )
            for individuo in poblacion
        ]
        
        mejor_aptitud_actual = max(aptitudes)
        if mejor_aptitud_actual > mejor_aptitud:
            mejor_aptitud = mejor_aptitud_actual
            mejor_individuo = copy.deepcopy(poblacion[aptitudes.index(mejor_aptitud_actual)])
            
        historial_aptitud.append(round(mejor_aptitud, 2))
        
    configuracion_genetico = {
        'tamano_poblacion': tamano_poblacion,
        'numero_generaciones': numero_generaciones,
        'metodo_seleccion': metodo_seleccion,
        'tamano_torneo': tamano_torneo,
        'probabilidad_cruce': probabilidad_cruce,
        'probabilidad_mutacion': probabilidad_mutacion,
        'elite_count': cantidad_elite
    }
    
    return mejor_individuo['cortes'], mejor_individuo['pesos'], historial_aptitud, configuracion_genetico

# -----------------------------------------------------------------------------
# Ejecucion directa de la Fase 4
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    import sys
    import matplotlib.pyplot as plt
    
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from Fase_1_Datos.preparacion import cargar_y_preparar
    from Fase_2_PRISM.prism import inducir_reglas_prism
    from Fase_3_Logica_Difusa.difuso import calcular_exactitud_dataset
    
    ruta_archivo = "winequality-red.csv"
    if not os.path.exists(ruta_archivo):
        ruta_archivo = os.path.join(os.path.dirname(__file__), "..", "winequality-red.csv")
        
    print("----------------------------------------------------------------------")
    print("FASE 4: CALIBRACION DIFUSA CON ALGORITMO GENETICO")
    print("----------------------------------------------------------------------")
    
    datos_vino = cargar_y_preparar(ruta_archivo)
    reglas_prism = inducir_reglas_prism(datos_vino)
    
    print("Parametros de ejecucion:")
    print(f"  Poblacion (N)            : {TAMANO_POBLACION}")
    print(f"  Generaciones             : {NUMERO_GENERACIONES}")
    print(f"  Seleccion                : {METODO_SELECCION} (k={TAMANO_TORNEO})")
    print(f"  Probabilidad Cruce (Pc)  : {PROBABILIDAD_CRUCE * 100:.1f}%")
    print(f"  Probabilidad Mutacion(Pm): {PROBABILIDAD_MUTACION * 100:.1f}%")
    print(f"  Elitismo                 : {CANTIDAD_ELITE} individuos")
    
    print("\nEjecutando optimizacion evolutiva...")
    cortes_opt, pesos_opt, historial, config = optimizar_con_algoritmo_genetico(datos_vino, reglas_prism)
    
    exactitud_entrenamiento = calcular_exactitud_dataset(datos_vino['df_train'], cortes_opt, reglas_prism, pesos_opt)
    exactitud_prueba = calcular_exactitud_dataset(datos_vino['df_test'], cortes_opt, reglas_prism, pesos_opt)
    reglas_utiles = sum(1 for peso in pesos_opt if peso > 0.1)
    
    print(f"\nExactitud Entrenamiento : {exactitud_entrenamiento}% (Inicial: {historial[0]}%)")
    print(f"Exactitud Prueba (Test) : {exactitud_prueba}%")
    print(f"Reglas activas          : {reglas_utiles} de {len(reglas_prism)}")
    
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(historial)), historial, marker='o', color='#c0392b', linewidth=2, markersize=4)
    plt.title("Convergencia del Fitness - Algoritmo Genetico")
    plt.xlabel("Generacion")
    plt.ylabel("Exactitud (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    print("\nMostrando grafico...")
    plt.show()
