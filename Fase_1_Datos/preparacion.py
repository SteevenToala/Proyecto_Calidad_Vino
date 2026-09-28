# Proyecto Inteligencia Artificial
# Fase 1: Carga, Justificacion Enologica y Discretizacion por Terciles

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Nombres en español de las 11 variables fisicoquimicas
MAPA_VARIABLES = {
    'fixed acidity': 'acidez_fija',
    'volatile acidity': 'acidez_volatil',
    'citric acid': 'acido_citrico',
    'residual sugar': 'azucar_residual',
    'chlorides': 'cloruros',
    'free sulfur dioxide': 'dioxido_azufre_libre',
    'total sulfur dioxide': 'dioxido_azufre_total',
    'density': 'densidad',
    'pH': 'ph',
    'sulphates': 'sulfatos',
    'alcohol': 'alcohol'
}

# Justificacion quimica de cada variable en la calidad del vino
JUSTIFICACIONES_QUIMICAS = {
    'alcohol': {
        'nombre': 'Graduacion Alcoholica (% vol.)',
        'impacto': 'Fuerte Positivo (+)',
        'justificacion': 'Aporta calidez, volumen y estructura en boca; es el principal factor en vinos de alta calidad.'
    },
    'acidez_volatil': {
        'nombre': 'Acidez Volatil (g/dm3 ac. acetico)',
        'impacto': 'Fuerte Negativo (-)',
        'justificacion': 'Genera notas agrias y sabor a vinagre por degradacion bacteriana. Defecto critico en vinos bajos.'
    },
    'sulfatos': {
        'nombre': 'Sulfatos (g/dm3 sulfato potasio)',
        'impacto': 'Moderado Positivo (+)',
        'justificacion': 'Antioxidante y conservante que preserva la frescura y aromas frutales.'
    },
    'acido_citrico': {
        'nombre': 'Acido Citrico (g/dm3)',
        'impacto': 'Moderado Positivo (+)',
        'justificacion': 'Añade frescura citrica y balance gustativo, evitando sensaciones empalagosas.'
    },
    'acidez_fija': {
        'nombre': 'Acidez Fija (g/dm3 tartarico)',
        'impacto': 'Leve Positivo (+)',
        'justificacion': 'Estructura acida base natural de la uva que da frescura y vivacidad.'
    },
    'densidad': {
        'nombre': 'Densidad (g/cm3)',
        'impacto': 'Moderado Negativo (-)',
        'justificacion': 'Relacionada con el balance azucar/alcohol (a mayor alcohol menor densidad).'
    },
    'total sulfur dioxide': {
        'nombre': 'Dioxido Azufre Total (mg/dm3)',
        'impacto': 'Moderado Negativo (-)',
        'justificacion': 'Suma de SO2 libre y ligado; concentraciones elevadas opacan el aroma frutal.'
    },
    'chlorides': {
        'nombre': 'Cloruros (g/dm3 NaCl)',
        'impacto': 'Leve Negativo (-)',
        'justificacion': 'Contenido mineral salino; en exceso produce aspereza en boca.'
    },
    'ph': {
        'nombre': 'Nivel de pH',
        'impacto': 'Regulador Estructural',
        'justificacion': 'Determina la acidez activa y la estabilidad biologica (tipico entre 3.0 y 3.8).'
    },
    'free sulfur dioxide': {
        'nombre': 'Dioxido Azufre Libre (mg/dm3)',
        'impacto': 'Regulador Activo',
        'justificacion': 'Fraccion activa de SO2 que previene oxidacion y proliferacion bacteriana.'
    },
    'azucar_residual': {
        'nombre': 'Azucar Residual (g/dm3)',
        'impacto': 'Secundario',
        'justificacion': 'Cantidad de azucar fermentable remanente tras la fermentacion.'
    }
}

# Carga de datos, calculo de Pearson, particion 80/20 y discretizacion
def cargar_y_preparar(ruta_archivo_csv):
    # Cargar archivo csv delimitado por punto y coma
    archivo_crudo = pd.read_csv(ruta_archivo_csv, sep=";")
    datos_vinos = archivo_crudo.rename(columns=MAPA_VARIABLES)
    
    # Calcular correlacion de Pearson de cada variable con la calidad original
    correlaciones_calidad = archivo_crudo.corr()['quality']
    informacion_variables = {}
    
    for nombre_original, nombre_espanol in MAPA_VARIABLES.items():
        coeficiente = round(float(correlaciones_calidad[nombre_original]), 4)
        info = JUSTIFICACIONES_QUIMICAS.get(nombre_original, JUSTIFICACIONES_QUIMICAS.get(nombre_espanol, {}))
        informacion_variables[nombre_espanol] = {
            'columna_original': nombre_original,
            'nombre_formal': info.get('nombre', nombre_espanol),
            'correlacion': coeficiente,
            'impacto': info.get('impacto', 'Neutro'),
            'justificacion': info.get('justificacion', ''),
            'minimo': round(float(datos_vinos[nombre_espanol].min()), 3),
            'promedio': round(float(datos_vinos[nombre_espanol].mean()), 3),
            'maximo': round(float(datos_vinos[nombre_espanol].max()), 3)
        }
        
    # Categorizar en 3 clases: BAJA (<=5), MEDIA (6), ALTA (>=7)
    def asignar_clase_calidad(puntaje):
        if puntaje <= 5:
            return 'BAJA'
        elif puntaje == 6:
            return 'MEDIA'
        else:
            return 'ALTA'
            
    datos_vinos['calidad'] = archivo_crudo['quality'].apply(asignar_clase_calidad)
    
    # Particion estratificada 80% entrenamiento y 20% prueba
    datos_entrenamiento, datos_prueba = train_test_split(
        datos_vinos, test_size=0.20, random_state=42, stratify=datos_vinos['calidad']
    )
    datos_entrenamiento = datos_entrenamiento.copy()
    datos_prueba = datos_prueba.copy()
    
    # Discretizacion en terciles estadisticos sobre el conjunto de entrenamiento
    cortes_base = {}
    atributos_quimicos = list(MAPA_VARIABLES.values())
    
    for atributo in atributos_quimicos:
        valor_minimo = float(datos_entrenamiento[atributo].min())
        tercil_bajo = float(datos_entrenamiento[atributo].quantile(0.333))
        valor_mediana = float(datos_entrenamiento[atributo].median())
        tercil_alto = float(datos_entrenamiento[atributo].quantile(0.667))
        valor_maximo = float(datos_entrenamiento[atributo].max())
        
        cortes_base[atributo] = {
            'minimo': round(valor_minimo, 4),
            'corte_bajo': round(tercil_bajo, 4),
            'punto_medio': round(valor_mediana, 4),
            'corte_alto': round(tercil_alto, 4),
            'maximo': round(valor_maximo, 4)
        }
        
        def asignar_etiqueta_discreta(valor, limite_bajo=tercil_bajo, limite_alto=tercil_alto):
            if valor <= limite_bajo:
                return 'bajo'
            elif valor <= limite_alto:
                return 'medio'
            else:
                return 'alto'
                
        datos_entrenamiento[atributo + '_discreto'] = datos_entrenamiento[atributo].apply(asignar_etiqueta_discreta)
        datos_prueba[atributo + '_discreto'] = datos_prueba[atributo].apply(asignar_etiqueta_discreta)
        
    return {
        'datos_completos': datos_vinos,
        'datos_entrenamiento': datos_entrenamiento,
        'datos_prueba': datos_prueba,
        'informacion_variables': informacion_variables,
        'cortes_base': cortes_base,
        'columnas_quimicas': atributos_quimicos,
        'df_completo': datos_vinos,
        'df_train': datos_entrenamiento,
        'df_test': datos_prueba,
        'info_variables': informacion_variables,
        'cols_quimicas': atributos_quimicos
    }

# Ejecucion directa de la Fase 1
if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    
    ruta_archivo = "winequality-red.csv"
    if not os.path.exists(ruta_archivo):
        ruta_archivo = os.path.join(os.path.dirname(__file__), "..", "winequality-red.csv")
        
    print("\nFase 1: Preparacion de Datos y Justificacion Enologica")
    datos_preparados = cargar_y_preparar(ruta_archivo)
    print(f"Total de registros cargados: {len(datos_preparados['datos_completos'])}")
    print(f"Muestras de entrenamiento  : {len(datos_preparados['datos_entrenamiento'])}")
    print(f"Muestras de prueba (test)  : {len(datos_preparados['datos_prueba'])}\n")
    
    print("Correlacion de Pearson con la Calidad:")
    for atributo in datos_preparados['columnas_quimicas']:
        info = datos_preparados['informacion_variables'][atributo]
        print(f"  - {atributo:<22}: r = {info['correlacion']:+0.4f} ({info['impacto']})")
        
    conteo_clases = datos_preparados['datos_completos']['calidad'].value_counts()[['BAJA', 'MEDIA', 'ALTA']]
    plt.figure(figsize=(6, 4))
    plt.bar(['BAJA', 'MEDIA', 'ALTA'], conteo_clases.values, color=['#c0392b', '#f39c12', '#27ae60'], edgecolor='black', alpha=0.85)
    plt.title("Distribucion de Clases de Calidad")
    plt.xlabel("Clase de Calidad")
    plt.ylabel("Cantidad de Vinos")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    print("\nMostrando grafico de distribucion...")
    plt.show()
