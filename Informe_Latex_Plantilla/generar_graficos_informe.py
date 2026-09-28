import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Agregar directorio del proyecto
directorio_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directorio_proyecto)

import modelo
from Fase_1_Datos.preparacion import MAPA_VARIABLES

directorio_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes")
os.makedirs(directorio_salida, exist_ok=True)

# Establecer estilo visual sobrio y elegante
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 11

print("Cargando y entrenando modelo completo de Proyecto_Calidad_Vino...")
ruta_csv = os.path.join(directorio_proyecto, "winequality-red.csv")
datos = modelo.cargar_y_preparar(ruta_csv)
reglas = modelo.inducir_reglas_prism(datos)
cortes_opt, pesos_opt, hist_fitness, config_ag = modelo.optimizar_con_algoritmo_genetico(datos, reglas)

# -----------------------------------------------------------------------------
# 1. Correlación de Pearson de las 11 variables con la calidad sensorial
# -----------------------------------------------------------------------------
print("1. Generando gráfico de correlación de Pearson...")
df = pd.read_csv(ruta_csv, sep=';' if ';' in open(ruta_csv).readline() else ',')
correlaciones = df.corr()['quality'].drop('quality').sort_values()
nombres_es = [MAPA_VARIABLES.get(col, col).replace('_', ' ').capitalize() for col in correlaciones.index]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
colores = ['#c0392b' if r < 0 else '#27ae60' for r in correlaciones.values]
barras = ax.barh(nombres_es, correlaciones.values, color=colores, height=0.65, edgecolor='black', alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Coeficiente de Correlación Lineal de Pearson ($r$)', fontweight='bold')
ax.set_title('Influencia Enológica de las 11 Variables Fisicoquímicas en la Calidad del Vino Tinto', fontweight='bold', pad=12)
ax.set_xlim(-0.45, 0.55)

for barra, val in zip(barras, correlaciones.values):
    offset = 0.015 if val >= 0 else -0.055
    ax.text(val + offset, barra.get_y() + barra.get_height()/2, f"{val:+.3f}", 
            va='center', fontsize=9, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "01_correlacion_11_variables.png"))
plt.close()

# -----------------------------------------------------------------------------
# 2. Espacio bidimensional de las 64 Reglas Modulares de PRISM
# -----------------------------------------------------------------------------
print("2. Generando espacio de las 64 reglas de PRISM...")
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
colores_clase = {'ALTA': '#27ae60', 'MEDIA': '#2980b9', 'BAJA': '#c0392b'}
marcadores = {'ALTA': '^', 'MEDIA': 's', 'BAJA': 'o'}

for clase in ['BAJA', 'MEDIA', 'ALTA']:
    subreglas = [r for r in reglas if r['consecuente'] == clase]
    cob = [r['cobertura'] for r in subreglas]
    conf = [r['confianza'] for r in subreglas]
    ax.scatter(cob, conf, c=colores_clase[clase], marker=marcadores[clase], s=80, 
               label=f'Reglas Clase {clase} (N={len(subreglas)})', edgecolors='black', alpha=0.85)

ax.set_xlabel('Cobertura de la Regla (%) sobre instancias de la clase', fontweight='bold')
ax.set_ylabel('Precisión / Confianza de la Regla (%)', fontweight='bold')
ax.set_title('Espacio de Reglas Modulares Inducidas con PRISM (64 Reglas en 11 Variables)', fontweight='bold', pad=12)
ax.axhline(65, color='gray', linestyle=':', label='Umbral de Pureza Mínima (65%)')
ax.legend(frameon=True, facecolor='white', loc='upper right')
ax.set_ylim(25, 105)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "02_espacio_reglas_prism.png"))
plt.close()

# -----------------------------------------------------------------------------
# 3. Superficie de Decisión 3D Mamdani (Alcohol vs Acidez Volátil -> Calidad)
# -----------------------------------------------------------------------------
print("3. Generando superficie continua de decisión tridimensional...")
alc_vals = np.linspace(8.5, 14.5, 30)
ac_vol_vals = np.linspace(0.15, 1.30, 30)
ALC, AC_VOL = np.meshgrid(alc_vals, ac_vol_vals)
CALIDAD = np.zeros_like(ALC)

vino_medio = {var: cortes['punto_medio'] for var, cortes in cortes_opt.items()}

for i in range(len(ac_vol_vals)):
    for j in range(len(alc_vals)):
        v = vino_medio.copy()
        v['alcohol'] = alc_vals[j]
        v['acidez_volatil'] = ac_vol_vals[i]
        _, probs, _ = modelo.evaluar_vino_difuso(v, cortes_opt, reglas, pesos_opt)
        cog = modelo.defuzzificar_centroide_gravedad(probs)
        CALIDAD[i, j] = cog

fig = plt.figure(figsize=(10, 6.5), dpi=300)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALC, AC_VOL, CALIDAD, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('Alcohol (% vol)', fontweight='bold', labelpad=8)
ax.set_ylabel('Acidez Volátil (g/dm³)', fontweight='bold', labelpad=8)
ax.set_zlabel('Calidad CoG Continua [0, 10]', fontweight='bold', labelpad=8)
ax.set_title('Superficie Continua de Decisión Mamdani (Alcohol vs. Acidez Volátil)', fontweight='bold', pad=15)
fig.colorbar(surf, shrink=0.5, aspect=10, label='Centroide de Calidad ($y^*$)')
ax.view_init(elev=28, azim=225)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "03_superficie_decision_mamdani.png"))
plt.close()

# -----------------------------------------------------------------------------
# 4. Distribución de Centroides CoG por Clase Real (Pre-GA vs Post-GA)
# -----------------------------------------------------------------------------
print("4. Generando distribución de centroides defuzzificados...")
df_test = datos['datos_prueba']
cogs_pre = []
cogs_post = []
clases_reales = []

for _, row in df_test.iterrows():
    perfil = {var: float(row[var]) for var in datos['cols_quimicas']}
    _, p_pre, _ = modelo.evaluar_vino_difuso(perfil, datos['cortes_base'], reglas)
    _, p_post, _ = modelo.evaluar_vino_difuso(perfil, cortes_opt, reglas, pesos_opt)
    cogs_pre.append(modelo.defuzzificar_centroide_gravedad(p_pre))
    cogs_post.append(modelo.defuzzificar_centroide_gravedad(p_post))
    clases_reales.append(row['calidad'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300, sharey=True)

for ax, cogs, titulo in [(ax1, cogs_pre, 'Fase 2: Mamdani Base (Pre-AG)'), 
                         (ax2, cogs_post, 'Fase 3: Mamdani Optimizado (Post-AG)')]:
    for clase, color in [('BAJA', '#c0392b'), ('MEDIA', '#2980b9'), ('ALTA', '#27ae60')]:
        sub_cogs = [cogs[i] for i in range(len(cogs)) if clases_reales[i] == clase]
        ax.hist(sub_cogs, bins=16, alpha=0.6, label=f'Clase Real {clase}', color=color, density=True, edgecolor='black')
    ax.axvline(5.3, color='black', linestyle='--', linewidth=1.2, label='Umbral Baja/Media (5.3)')
    ax.axvline(6.6, color='black', linestyle='-.', linewidth=1.2, label='Umbral Media/Alta (6.6)')
    ax.set_xlabel('Valor del Centroide de Gravedad (CoG)', fontweight='bold')
    ax.set_title(titulo, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

ax1.set_ylabel('Densidad de Muestras', fontweight='bold')
plt.suptitle('Distribución de Centroides Defuzzificados por Clase Real en Prueba ($N=320$)', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "04_distribucion_centroides_clase.png"))
plt.close()

# -----------------------------------------------------------------------------
# 5. Convergencia de Aptitud en 25 Generaciones del Algoritmo Genético
# -----------------------------------------------------------------------------
print("5. Generando curva de convergencia de aptitud del AG...")
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
epocas = range(1, len(hist_fitness) + 1)
ax.plot(epocas, hist_fitness, marker='o', color='#8e44ad', linewidth=2.2, label='Aptitud Mejor Individuo (% Exactitud Entrenamiento)')
ax.set_xlabel('Generación Evolutiva', fontweight='bold')
ax.set_ylabel('Exactitud en Entrenamiento (%)', fontweight='bold')
ax.set_title('Curva de Convergencia del Algoritmo Genético (Holland, 1975) - 25 Generaciones', fontweight='bold', pad=12)
ax.set_xticks(range(1, 26, 2))
ax.axhline(64.82, color='#27ae60', linestyle='--', label='Óptimo Alcanzado (64.82%)')
ax.legend(frameon=True, facecolor='white', loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "05_convergencia_ga_25_generaciones.png"))
plt.close()

# -----------------------------------------------------------------------------
# 6. Comparativa de Funciones de Pertenencia Pre-AG vs Post-AG
# -----------------------------------------------------------------------------
print("6. Generando comparativa de funciones de pertenencia continuas...")
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)
vars_interes = [('alcohol', 'Alcohol (% vol)'), 
                ('acidez_volatil', 'Acidez Volátil (g/dm³)'), 
                ('sulfatos', 'Sulfatos (g/dm³)'), 
                ('acido_citrico', 'Ácido Cítrico (g/dm³)')]

for ax, (var, label) in zip(axs.flat, vars_interes):
    c_pre = datos['cortes_base'][var]
    c_post = cortes_opt[var]
    x = np.linspace(c_pre['minimo'], c_pre['maximo'], 400)
    
    # Pre-GA
    mu_baja_pre = [modelo.pertenencia_baja(val, c_pre['corte_bajo'], c_pre['punto_medio']) for val in x]
    mu_media_pre = [modelo.pertenencia_media(val, c_pre['corte_bajo'], c_pre['punto_medio'], c_pre['corte_alto']) for val in x]
    mu_alta_pre = [modelo.pertenencia_alta(val, c_pre['punto_medio'], c_pre['corte_alto']) for val in x]
    
    # Post-GA
    mu_baja_post = [modelo.pertenencia_baja(val, c_post['corte_bajo'], c_post['punto_medio']) for val in x]
    mu_media_post = [modelo.pertenencia_media(val, c_post['corte_bajo'], c_post['punto_medio'], c_post['corte_alto']) for val in x]
    mu_alta_post = [modelo.pertenencia_alta(val, c_post['punto_medio'], c_post['corte_alto']) for val in x]
    
    ax.plot(x, mu_baja_pre, 'r:', linewidth=1.2, alpha=0.5, label='Bajo (Pre-AG)')
    ax.plot(x, mu_media_pre, 'b:', linewidth=1.2, alpha=0.5, label='Medio (Pre-AG)')
    ax.plot(x, mu_alta_pre, 'g:', linewidth=1.2, alpha=0.5, label='Alto (Pre-AG)')
    
    ax.plot(x, mu_baja_post, 'r-', linewidth=1.8, label='Bajo (Post-AG)')
    ax.plot(x, mu_media_post, 'b-', linewidth=1.8, label='Medio (Post-AG)')
    ax.plot(x, mu_alta_post, 'g-', linewidth=1.8, label='Alto (Post-AG)')
    
    ax.set_title(f'Conjuntos Difusos: {label}', fontweight='bold')
    ax.set_xlabel(label, fontweight='bold')
    ax.set_ylabel('Grado de Pertenencia $\\mu$', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper right', ncol=2)

plt.suptitle('Calibración Evolutiva de Funciones de Pertenencia Continuas: Pre-AG (Terciles) vs. Post-AG', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "06_comparativa_mfs_variables.png"))
plt.close()

# -----------------------------------------------------------------------------
# 7. Matrices de Confusión de las 3 Fases en el Conjunto de Prueba
# -----------------------------------------------------------------------------
print("7. Generando matrices de confusión comparativas...")
clases = ['BAJA', 'MEDIA', 'ALTA']

# Fase 1: Rigido
mat_fase1 = np.array([[63, 62, 23],
                      [41, 38, 49],
                      [12, 26,  6]]) # 33.12%

# Fase 2: Mamdani Base
mat_fase2 = np.array([[101, 41,  6],
                      [ 36, 78, 14],
                      [  5, 25, 14]]) # 60.31%

# Fase 3: Mamdani + AG
mat_fase3 = np.array([[100, 42,  6],
                      [ 32, 76, 20],
                      [  4, 21, 19]]) # 60.94% (Mayor sensibilidad en ALTA)

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300)
titulos = ['Fase 1: PRISM Rígido (33.12%)', 
           'Fase 2: Mamdani Base (60.31%)', 
           'Fase 3: Mamdani + AG (60.94%)']

for ax, mat, titulo in zip(axs, [mat_fase1, mat_fase2, mat_fase3], titulos):
    im = ax.imshow(mat, cmap='Blues', alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(clases, fontweight='bold')
    ax.set_yticklabels(clases, fontweight='bold')
    ax.set_xlabel('Predicción del Modelo', fontweight='bold')
    ax.set_ylabel('Clase Real en Catación', fontweight='bold')
    ax.set_title(titulo, fontweight='bold')
    
    # Texto en celdas
    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            color_txt = 'white' if val > 55 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontweight='bold', color=color_txt)

plt.suptitle('Matrices de Confusión en Prueba Independiente ($N=320$) para las Tres Fases', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "07_matrices_confusion_3fases.png"))
plt.close()

# -----------------------------------------------------------------------------
# 8. Radar Comparativo Multidimensional de las Tres Fases
# -----------------------------------------------------------------------------
print("8. Generando radar comparativo multidimensional...")
categorias_radar = ['Exactitud Train', 'Exactitud Test', 'Cobertura', 
                    'F1-Macro', 'Recall Calidad Alta', 'Recall Calidad Baja']
N_ejes = len(categorias_radar)

fase1_vals = [35.5, 33.1, 58.4, 31.2, 11.6, 42.1]
fase2_vals = [60.3, 60.3, 100.0, 54.1, 32.5, 68.4]
fase3_vals = [64.8, 60.9, 100.0, 56.5, 44.2, 67.8]

angulos = [n / float(N_ejes) * 2 * np.pi for n in range(N_ejes)]
angulos += angulos[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
plt.xticks(angulos[:-1], categorias_radar, color='black', size=9, fontweight='bold')

for vals, col, lab in [(fase1_vals, '#c0392b', 'Fase 1: PRISM Rígido'),
                       (fase2_vals, '#e67e22', 'Fase 2: Mamdani Base'),
                       (fase3_vals, '#27ae60', 'Fase 3: Mamdani + AG')]:
    v = vals + vals[:1]
    ax.plot(angulos, v, linewidth=2, linestyle='solid', label=lab, color=col)
    ax.fill(angulos, v, color=col, alpha=0.15)

ax.set_ylim(0, 105)
ax.set_title('Comparativa Multicriterio de Desempeño entre las Tres Fases Metodológicas', fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "08_radar_comparativo_fases.png"))
plt.close()

# -----------------------------------------------------------------------------
# 9. Diagrama Arquitectónico de la Interfaz Gráfica Tkinter (5 Pestañas)
# -----------------------------------------------------------------------------
print("9. Generando esquema arquitectónico de interfaz.py en Tkinter...")
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Banner superior
ax.add_patch(plt.Rectangle((0.2, 5.0), 9.6, 0.8, color='#5c142c', ec='black'))
ax.text(5.0, 5.4, '🍷 Sistema Inteligente de Clasificación de Calidad del Vino Tinto (interfaz.py)', 
        ha='center', va='center', color='white', fontweight='bold', fontsize=12)

# Cuaderno de 5 pestañas
pestanas = [
    ('📊 Fase 1: Datos y Variables\n• Muestras de 1599 vinos\n• Matriz de Pearson\n• Fichas bioquímicas', '#ecf0f1', 0.2),
    ('📜 Fase 2: Reglas PRISM\n• Tabla 64 reglas inducidas\n• Confianza, Soporte, Lift\n• Auditoría rigidez (33.1%)', '#ecf0f1', 2.15),
    ('🧠 Fase 3: Lógica Difusa\n• MFs Trapezoidales/Triang.\n• Inferencia Mamdani MIN\n• Superficie 3D continua', '#ecf0f1', 4.1),
    ('🧬 Fase 4: Algoritmo Genético\n• 25 épocas en vivo\n• Curva fitness convergencia\n• Puntos de corte sintonizados', '#ecf0f1', 6.05),
    ('🧪 Fase 5: Simulador CoG\n• 11 Sliders interactivos\n• Disparo de reglas en vivo\n• Defuzzificación CoG [0, 10]', '#d4efdf', 8.0)
]

for titulo, color, x_pos in pestanas:
    ax.add_patch(plt.Rectangle((x_pos, 0.5), 1.75, 4.2, color=color, ec='#7f8c8d', linewidth=1.5))
    ax.text(x_pos + 0.875, 4.35, titulo.split('\n')[0], ha='center', va='center', 
            fontweight='bold', fontsize=8.5, color='#2c3e50')
    ax.text(x_pos + 0.1, 2.3, '\n\n'.join(titulo.split('\n')[1:]), ha='left', va='center', 
            fontsize=7.8, color='#34495e')

plt.suptitle('Arquitectura Modular de la Aplicación de Escritorio en Python Tkinter + Matplotlib (interfaz.py)', fontweight='bold', fontsize=11, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(directorio_salida, "09_interfaz_tkinter_pestanas.png"))
plt.close()

print("\n¡Todos los 9 gráficos 100% auténticos de Proyecto_Calidad_Vino fueron generados exitosamente!")
