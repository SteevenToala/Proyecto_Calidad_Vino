# Proyecto Inteligencia Artificial
# Punto de entrada principal para ejecucion por consola o interfaz grafica
# Uso:
#   python main.py             (Abre la aplicacion de escritorio nativa)
#   python main.py --consola   (Ejecuta las fases en terminal)

import os
import sys
import time
import modelo
import interfaz

def main():
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo_csv = os.path.join(carpeta_actual, "winequality-red.csv")
    
    if not os.path.exists(ruta_archivo_csv):
        print(f"Error: No se encontro el archivo '{ruta_archivo_csv}'")
        sys.exit(1)
        
    print("\nProyecto Inteligencia Artificial: Clasificacion de Calidad del Vino Tinto")
    print("Flujo: 11 Variables -> Reglas PRISM -> Logica Difusa Mamdani -> Algoritmo Genetico\n")
    
    tiempo_inicio = time.time()
    
    # Fase 1: Carga y preparacion de datos
    print("[1/4] Cargando dataset y calculando justificaciones de 11 variables...")
    datos_preparados = modelo.cargar_y_preparar(ruta_archivo_csv)
    print(f"      - {len(datos_preparados['datos_completos'])} vinos cargados (Entrenamiento: {len(datos_preparados['datos_entrenamiento'])}, Prueba: {len(datos_preparados['datos_prueba'])})")
    
    # Fase 2: Induccion de reglas con PRISM
    print("\n[2/4] Induciendo reglas modulares con PRISM...")
    reglas_aprendidas = modelo.inducir_reglas_prism(datos_preparados)
    exactitud_entrenamiento_rigido = modelo.evaluar_prism_rigido(datos_preparados['datos_entrenamiento'], reglas_aprendidas)
    exactitud_prueba_rigido = modelo.evaluar_prism_rigido(datos_preparados['datos_prueba'], reglas_aprendidas)
    print(f"      - {len(reglas_aprendidas)} reglas inducidas exitosamente.")
    print(f"      - PRISM rigido (Fronteras booleanas): Entrenamiento = {exactitud_entrenamiento_rigido}%, Prueba = {exactitud_prueba_rigido}%")
    
    # Fase 3: Logica difusa Mamdani inicial
    print("\n[3/4] Evaluando inferencia difusa Mamdani con cortes iniciales...")
    exactitud_entrenamiento_inicial = modelo.calcular_exactitud_dataset(
        datos_preparados['datos_entrenamiento'], datos_preparados['cortes_base'], reglas_aprendidas
    )
    exactitud_prueba_inicial = modelo.calcular_exactitud_dataset(
        datos_preparados['datos_prueba'], datos_preparados['cortes_base'], reglas_aprendidas
    )
    print(f"      - Exactitud Difusa Inicial: Entrenamiento = {exactitud_entrenamiento_inicial}%, Prueba = {exactitud_prueba_inicial}%")
    
    # Fase 4: Calibracion evolutiva con Algoritmo Genetico
    print("\n[4/4] Optimizando cortes difusos y pesos con Algoritmo Genetico (John Holland)...")
    cortes_optimizados, pesos_optimizados, historial_aptitud, configuracion_genetico = modelo.optimizar_con_algoritmo_genetico(
        datos_preparados, reglas_aprendidas
    )
    exactitud_entrenamiento_optimizada = modelo.calcular_exactitud_dataset(
        datos_preparados['datos_entrenamiento'], cortes_optimizados, reglas_aprendidas, pesos_optimizados
    )
    exactitud_prueba_optimizada = modelo.calcular_exactitud_dataset(
        datos_preparados['datos_prueba'], cortes_optimizados, reglas_aprendidas, pesos_optimizados
    )
    reglas_activas = sum(1 for peso in pesos_optimizados if peso > 0.1)
    
    print(f"      - Hiperparametros: Poblacion={configuracion_genetico['tamano_poblacion']}, Generaciones={configuracion_genetico['numero_generaciones']}, Seleccion={configuracion_genetico['metodo_seleccion'].upper()} (k={configuracion_genetico['tamano_torneo']})")
    print(f"      - Probabilidad de Cruce (Pc): {configuracion_genetico['probabilidad_cruce']*100:.1f}%, Probabilidad de Mutacion (Pm): {configuracion_genetico['probabilidad_mutacion']*100:.1f}%, Elite={configuracion_genetico['elite_count']}")
    print(f"      - Exactitud Optimizada con AG: Entrenamiento = {exactitud_entrenamiento_optimizada}%, Prueba = {exactitud_prueba_optimizada}%")
    print(f"      - Reglas activas calibradas: {reglas_activas} de {len(reglas_aprendidas)}")
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\nEntrenamiento secuencial completado en {tiempo_total:.2f} segundos.\n")
    
    modelo_completo = {
        'datos': datos_preparados,
        'reglas': reglas_aprendidas,
        'cortes_base': datos_preparados['cortes_base'],
        'cortes_optimizados': cortes_optimizados,
        'pesos_optimizados': pesos_optimizados,
        'exactitud_entrenamiento_rigido': exactitud_entrenamiento_rigido,
        'exactitud_prueba_rigido': exactitud_prueba_rigido,
        'exactitud_entrenamiento_inicial': exactitud_entrenamiento_inicial,
        'exactitud_entrenamiento_optimizada': exactitud_entrenamiento_optimizada,
        'exactitud_prueba_inicial': exactitud_prueba_inicial,
        'exactitud_prueba_optimizada': exactitud_prueba_optimizada,
        'reglas_activas': reglas_activas,
        'total_reglas': len(reglas_aprendidas),
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
    
    # Si se pasa --consola solo muestra la salida en terminal
    if len(sys.argv) > 1 and sys.argv[1] == '--consola':
        print("\nEjecucion en modo consola finalizada.")
        return
        
    # Por defecto, abrir la ventana nativa de escritorio
    print("\nAbriendo interfaz grafica de escritorio (Tkinter)...")
    interfaz.iniciar_interfaz(modelo_completo)

if __name__ == "__main__":
    main()
