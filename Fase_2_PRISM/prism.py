# Proyecto Inteligencia Artificial
# Fase 2: Induccion de Reglas de Clasificacion con PRISM (J. Cendrowska, 1987)

import pandas as pd

# Procedimiento Recubrimiento_secuencial (Clases, atributos, ejemplos)
def recubrimiento_secuencial_prism(datos_preparados):
    datos_entrenamiento = datos_preparados['df_train']
    atributos_quimicos = datos_preparados['cols_quimicas']
    total_muestras_entrenamiento = len(datos_entrenamiento)
    clases_calidad = ['ALTA', 'BAJA', 'MEDIA']
    
    # Probabilidad previa P(C) de cada clase para el calculo del Lift
    probabilidades_previas = {
        clase: (datos_entrenamiento['calidad'] == clase).sum() / total_muestras_entrenamiento
        for clase in clases_calidad
    }
    columnas_discretizadas = [columna + '_discreto' for columna in atributos_quimicos]
    
    reglas_inducidas = []
    identificador_regla = 1
    
    # Para cada clase C de Clases
    for clase in clases_calidad:
        ejemplos_restantes = datos_entrenamiento.copy()
        
        # Mientras E contenga ejemplos de la clase C
        while True:
            positivos_restantes = (ejemplos_restantes['calidad'] == clase).sum()
            if positivos_restantes < 5:
                break
                
            # Aprender una regla para la clase actual
            regla, filtro_cobertura, aciertos = aprender_una_regla(
                clase,
                ejemplos_restantes,
                datos_entrenamiento,
                columnas_discretizadas,
                probabilidades_previas[clase]
            )
            
            if regla is None or aciertos < 3:
                break
                
            regla['id'] = identificador_regla
            identificador_regla += 1
            reglas_inducidas.append(regla)
            
            # Principio de recubrimiento: E = E - {ejemplos cubiertos por Regla de la clase C}
            filtro_local = filtro_cobertura.loc[ejemplos_restantes.index]
            indices_cubiertos = ejemplos_restantes[filtro_local & (ejemplos_restantes['calidad'] == clase)].index
            ejemplos_restantes = ejemplos_restantes.drop(index=indices_cubiertos, errors='ignore')
            
            if len(indices_cubiertos) == 0:
                break
                
    return reglas_inducidas

# Procedimiento AprenderUnaRegla (Clase, Ejemplos, Atributos)
def aprender_una_regla(clase_objetivo, ejemplos_disponibles, datos_entrenamiento, atributos_discretos, probabilidad_previa_clase):
    antecedente = {}
    atributos_libres = list(atributos_discretos)
    ejemplos_iteracion = ejemplos_disponibles.copy()
    etiquetas_posibles = ['bajo', 'medio', 'alto']
    
    # MIENTRAS la regla cubra ejemplos negativos y queden atributos disponibles
    while atributos_libres and len(antecedente) < 3:
        restricciones_candidatas = [
            (atributo, etiqueta)
            for atributo in atributos_libres
            for etiqueta in etiquetas_posibles
        ]
        
        # Seleccionar la mejor restriccion por Confianza P(B|A)
        restriccion_optima, confianza, cobertura = mejor_restriccion(
            restricciones_candidatas, ejemplos_iteracion, clase_objetivo
        )
        
        if restriccion_optima is None or cobertura < 5:
            break
            
        columna_seleccionada, etiqueta_seleccionada = restriccion_optima
        nombre_variable_real = columna_seleccionada.replace('_discreto', '')
        
        antecedente[nombre_variable_real] = etiqueta_seleccionada
        atributos_libres.remove(columna_seleccionada)
        
        ejemplos_iteracion = ejemplos_iteracion[ejemplos_iteracion[columna_seleccionada] == etiqueta_seleccionada]
        
        # Condicion de pureza o confianza aceptable
        ejemplos_negativos = (ejemplos_iteracion['calidad'] != clase_objetivo).sum()
        if ejemplos_negativos == 0 or confianza >= 0.65:
            break
            
    if not antecedente:
        return None, None, 0
        
    # Calculo formal de metricas: Confianza, Soporte, Cobertura y Lift
    filtro_antecedente = pd.Series(True, index=datos_entrenamiento.index)
    for variable, etiqueta in antecedente.items():
        filtro_antecedente &= (datos_entrenamiento[variable + '_discreto'] == etiqueta)
        
    total_cumplen_antecedente = filtro_antecedente.sum()
    ejemplos_correctos = (filtro_antecedente & (datos_entrenamiento['calidad'] == clase_objetivo)).sum()
    total_muestras_entrenamiento = len(datos_entrenamiento)
    total_positivos_clase = (datos_entrenamiento['calidad'] == clase_objetivo).sum()
    
    if total_cumplen_antecedente == 0:
        return None, None, 0
        
    confianza_regla = ejemplos_correctos / total_cumplen_antecedente
    soporte_regla = ejemplos_correctos / total_muestras_entrenamiento
    cobertura_regla = ejemplos_correctos / total_positivos_clase if total_positivos_clase > 0 else 0.0
    lift_regla = confianza_regla / probabilidad_previa_clase if probabilidad_previa_clase > 0 else 1.0
    
    regla_aprendida = {
        'condiciones': antecedente,
        'consecuente': clase_objetivo,
        'confianza': round(float(confianza_regla * 100.0), 1),
        'precision': round(float(confianza_regla * 100.0), 1),
        'soporte': round(float(soporte_regla * 100.0), 1),
        'cobertura': round(float(cobertura_regla * 100.0), 1),
        'lift': round(float(lift_regla), 2),
        'aciertos': int(ejemplos_correctos),
        'cubiertos': int(total_cumplen_antecedente)
    }
    
    return regla_aprendida, filtro_antecedente, ejemplos_correctos

# Procedimiento mejorRestriccion (Restricciones, regla)
def mejor_restriccion(restricciones_candidatas, ejemplos_actuales, clase_objetivo):
    restriccion_seleccionada = None
    confianza_maxima = -1.0
    cobertura_maxima = -1
    
    for columna_discretizada, etiqueta_valor in restricciones_candidatas:
        subconjunto_ejemplos = ejemplos_actuales[ejemplos_actuales[columna_discretizada] == etiqueta_valor]
        total_subconjunto = len(subconjunto_ejemplos)
        
        if total_subconjunto == 0:
            continue
            
        ejemplos_correctos = (subconjunto_ejemplos['calidad'] == clase_objetivo).sum()
        confianza_actual = ejemplos_correctos / total_subconjunto
        
        # Criterio PRISM: mayor confianza, desempate por mayor cobertura
        if (confianza_actual > confianza_maxima) or (confianza_actual == confianza_maxima and ejemplos_correctos > cobertura_maxima):
            confianza_maxima = confianza_actual
            cobertura_maxima = ejemplos_correctos
            restriccion_seleccionada = (columna_discretizada, etiqueta_valor)
            
    return restriccion_seleccionada, confianza_maxima, cobertura_maxima

# Evaluacion booleana rigida para evidenciar el Boundary Problem
def evaluar_prism_rigido(datos_evaluacion, lista_reglas):
    aciertos = 0
    total_muestras = len(datos_evaluacion)
    
    for _, fila in datos_evaluacion.iterrows():
        clase_predicha = None
        for regla in lista_reglas:
            cumple_regla = True
            for variable, etiqueta in regla['condiciones'].items():
                if fila[variable + '_discreto'] != etiqueta:
                    cumple_regla = False
                    break
            if cumple_regla:
                clase_predicha = regla['consecuente']
                break
                
        if clase_predicha is None:
            clase_predicha = 'MEDIA'
            
        if clase_predicha == fila['calidad']:
            aciertos += 1
            
    exactitud = (aciertos / total_muestras) * 100.0
    return round(float(exactitud), 2)

# Alias para compatibilidad general
inducir_reglas_prism = recubrimiento_secuencial_prism

# Ejecucion independiente de la Fase 2
if __name__ == '__main__':
    import os
    import sys
    import matplotlib.pyplot as plt
    
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from Fase_1_Datos.preparacion import cargar_y_preparar
    
    ruta_archivo = "winequality-red.csv"
    if not os.path.exists(ruta_archivo):
        ruta_archivo = os.path.join(os.path.dirname(__file__), "..", "winequality-red.csv")
        
    print("----------------------------------------------------------------------")
    print("FASE 2: INDUCCION DE REGLAS CON ALGORITMO PRISM (J. CENDROWSKA, 1987)")
    print("----------------------------------------------------------------------")
    
    datos_preparados = cargar_y_preparar(ruta_archivo)
    reglas_aprendidas = recubrimiento_secuencial_prism(datos_preparados)
    print(f"Total de reglas modulares inducidas: {len(reglas_aprendidas)}")
    
    exactitud_entrenamiento = evaluar_prism_rigido(datos_preparados['df_train'], reglas_aprendidas)
    exactitud_prueba = evaluar_prism_rigido(datos_preparados['df_test'], reglas_aprendidas)
    print(f"Exactitud PRISM rigido en Entrenamiento : {exactitud_entrenamiento}%")
    print(f"Exactitud PRISM rigido en Prueba (Test)  : {exactitud_prueba}%")
    
    print("\nEjemplo de las primeras 3 reglas aprendidas:")
    for regla in reglas_aprendidas[:3]:
        condiciones_texto = " Y ".join([f"{var}={etiq}" for var, etiq in regla['condiciones'].items()])
        print(f"  Regla {regla['id']}: SI {condiciones_texto} ENTONCES {regla['consecuente']}")
        print(f"     [Confianza: {regla['confianza']}%, Soporte: {regla['soporte']}%, Cobertura: {regla['cobertura']}%, Lift: {regla['lift']}]")
        
    conteo_consecuente = {'ALTA': 0, 'MEDIA': 0, 'BAJA': 0}
    for regla in reglas_aprendidas:
        conteo_consecuente[regla['consecuente']] += 1
        
    plt.figure(figsize=(6.5, 4))
    plt.bar(list(conteo_consecuente.keys()), list(conteo_consecuente.values()), color=['#27ae60', '#f39c12', '#c0392b'], edgecolor='black', alpha=0.85)
    plt.title("Reglas PRISM por Clase Consecuente")
    plt.xlabel("Clase Consecuente")
    plt.ylabel("Cantidad de Reglas")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    print("\nMostrando grafico de distribucion de reglas...")
    plt.show()
