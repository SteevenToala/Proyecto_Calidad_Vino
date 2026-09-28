# Proyecto Inteligencia Artificial
# Interfaz Grafica de Escritorio (Tkinter + Matplotlib)
# Arquitectura modular organizada estrictamente por Fases de Aprendizaje

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import modelo

class InterfazVinoFases:
    def __init__(self, ventana_principal, modelo_completo):
        self.ventana = ventana_principal
        self.modelo = modelo_completo
        self.datos = modelo_completo['datos']
        self.reglas = modelo_completo['reglas']
        self.cortes_optimizados = modelo_completo.get('cortes_optimizados', modelo_completo.get('cortes_opt', {}))
        self.pesos_optimizados = modelo_completo.get('pesos_optimizados', modelo_completo.get('pesos_opt', []))
        self.cortes_opt = self.cortes_optimizados
        self.pesos_opt = self.pesos_optimizados
        
        self.ventana.title("Sistema Inteligente de Clasificacion de Vinos - Flujo por Fases")
        self.ventana.geometry("1220x780")
        self.ventana.minsize(1050, 700)
        
        # Paleta de colores sobria
        self.color_primario = "#5c142c"
        self.color_secundario = "#2c3e50"
        self.color_fondo = "#f8f9fa"
        self.ventana.configure(bg=self.color_fondo)
        
        self.estilo = ttk.Style()
        try:
            self.estilo.theme_use('clam')
        except:
            pass
            
        self.estilo.configure(".", background=self.color_fondo, font=("Segoe UI", 9))
        self.estilo.configure("TNotebook", background=self.color_fondo)
        self.estilo.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6])
        self.estilo.map("TNotebook.Tab", 
                        background=[("selected", "#ffffff")], 
                        foreground=[("selected", self.color_primario)])
        
        # Banner superior de identificacion del proyecto
        self.crear_banner_superior()
        
        # Organizador de pestanas por fases
        self.cuaderno_fases = ttk.Notebook(self.ventana)
        self.cuaderno_fases.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        
        self.pestana_fase1 = ttk.Frame(self.cuaderno_fases)
        self.pestana_fase2 = ttk.Frame(self.cuaderno_fases)
        self.pestana_fase3 = ttk.Frame(self.cuaderno_fases)
        self.pestana_fase4 = ttk.Frame(self.cuaderno_fases)
        self.pestana_fase5 = ttk.Frame(self.cuaderno_fases)
        
        self.cuaderno_fases.add(self.pestana_fase1, text="  📊 Fase 1: Datos y Variables  ")
        self.cuaderno_fases.add(self.pestana_fase2, text="  📜 Fase 2: Reglas PRISM  ")
        self.cuaderno_fases.add(self.pestana_fase3, text="  🧠 Fase 3: Logica Difusa  ")
        self.cuaderno_fases.add(self.pestana_fase4, text="  🧬 Fase 4: Algoritmo Genetico  ")
        self.cuaderno_fases.add(self.pestana_fase5, text="  🧪 Fase 5: Simulador y Defuzzificacion  ")
        
        # Construccion visual de cada fase
        self.construir_fase1()
        self.construir_fase2()
        self.construir_fase3()
        self.construir_fase4()
        self.construir_fase5()
        
    def crear_banner_superior(self):
        marco_banner = tk.Frame(self.ventana, bg=self.color_primario, height=62)
        marco_banner.pack(fill="x", side="top", pady=(0, 8))
        
        etiqueta_titulo = tk.Label(
            marco_banner, text="🍷 Clasificacion Inteligente de Calidad del Vino Tinto",
            font=("Segoe UI", 13, "bold"), fg="white", bg=self.color_primario
        )
        etiqueta_titulo.pack(anchor="w", padx=16, pady=(6, 0))
        
        etiqueta_subtitulo = tk.Label(
            marco_banner, 
            text="Pipeline Academico: 11 Variables Fisicoquimicas ➔ Induccion PRISM ➔ Inferencia Difusa Mamdani ➔ Optimizacion AG",
            font=("Segoe UI", 8), fg="#e8c2cc", bg=self.color_primario
        )
        etiqueta_subtitulo.pack(anchor="w", padx=16, pady=(0, 6))

    # Fase 1: Carga, analisis de 11 variables fisicoquimicas y correlacion de Pearson
    def construir_fase1(self):
        contenedor = ttk.Frame(self.pestana_fase1, padding=6)
        contenedor.pack(fill="both", expand=True)
        
        # Tabla superior con muestras del dataset
        marco_tabla = ttk.LabelFrame(contenedor, text=" Muestras del Dataset (1,599 vinos x 11 variables fisicoquimicas) ", padding=4)
        marco_tabla.pack(fill="x", pady=(0, 6))
        
        datos_completos = self.datos.get('datos_completos', self.datos.get('df_completo'))
        columnas_datos = tuple(datos_completos.columns[:12])
        tabla_datos = ttk.Treeview(marco_tabla, columns=columnas_datos, show="headings", height=4)
        for columna in columnas_datos:
            tabla_datos.heading(columna, text=columna)
            tabla_datos.column(columna, width=82, anchor="center")
            
        barra_desplazamiento = ttk.Scrollbar(marco_tabla, orient="vertical", command=tabla_datos.yview)
        tabla_datos.configure(yscrollcommand=barra_desplazamiento.set)
        tabla_datos.pack(side="left", fill="x", expand=True)
        barra_desplazamiento.pack(side="right", fill="y")
        
        for _, fila in datos_completos.head(50).iterrows():
            tabla_datos.insert("", "end", values=tuple(fila[col] for col in columnas_datos))
            
        # Panel inferior: Justificaciones quimicas y graficos estadisticos
        marco_inferior = ttk.Frame(contenedor)
        marco_inferior.pack(fill="both", expand=True)
        
        # Columna izquierda: Justificaciones quimicas y Pearson
        marco_justificaciones = ttk.LabelFrame(marco_inferior, text=" Analisis de las 11 Variables y Correlacion de Pearson ", padding=4)
        marco_justificaciones.pack(side="left", fill="both", expand=True, padx=(0, 6))
        
        columnas_justificaciones = ("Variable", "Nombre Enologico", "r (Calidad)", "Impacto")
        tabla_justificaciones = ttk.Treeview(marco_justificaciones, columns=columnas_justificaciones, show="headings", height=8)
        for columna in columnas_justificaciones:
            tabla_justificaciones.heading(columna, text=columna)
        tabla_justificaciones.column("Variable", width=120)
        tabla_justificaciones.column("Nombre Enologico", width=175)
        tabla_justificaciones.column("r (Calidad)", width=75, anchor="center")
        tabla_justificaciones.column("Impacto", width=110, anchor="center")
        
        informacion_variables = self.datos.get('informacion_variables', self.datos.get('info_variables', {}))
        for variable, info in informacion_variables.items():
            tabla_justificaciones.insert("", "end", values=(
                variable, info['nombre_formal'], f"{info['correlacion']:+0.4f}", info['impacto']
            ))
        tabla_justificaciones.pack(fill="both", expand=True)
        
        # Columna derecha: Graficos embebidos (Distribucion de clases y Correlaciones)
        marco_graficos = ttk.LabelFrame(marco_inferior, text=" Visualizacion Grafica de Datos ", padding=4)
        marco_graficos.pack(side="right", fill="both", expand=True)
        
        cuaderno_graficos = ttk.Notebook(marco_graficos)
        cuaderno_graficos.pack(fill="both", expand=True)
        
        # Grafico 1: Distribucion de clases de calidad
        pestana_distribucion = ttk.Frame(cuaderno_graficos)
        cuaderno_graficos.add(pestana_distribucion, text=" Distribucion de Calidad ")
        
        figura_distribucion, eje_distribucion = plt.subplots(figsize=(4.8, 3.2), dpi=92)
        conteo_clases = datos_completos['calidad'].value_counts()[['BAJA', 'MEDIA', 'ALTA']]
        colores_clases = ['#dc3545', '#ffc107', '#28a745']
        barras_clases = eje_distribucion.bar(['BAJA (<=5)', 'MEDIA (6)', 'ALTA (>=7)'], conteo_clases.values, color=colores_clases, edgecolor='black', alpha=0.85)
        eje_distribucion.set_ylabel("Muestras")
        eje_distribucion.set_title("Distribucion de Clases de Vino", fontsize=9, fontweight='bold')
        eje_distribucion.grid(axis='y', linestyle='--', alpha=0.4)
        for barra, valor in zip(barras_clases, conteo_clases.values):
            eje_distribucion.text(barra.get_x() + barra.get_width()/2, valor + 15, f"{valor}", ha='center', fontsize=8, fontweight='bold')
        figura_distribucion.tight_layout()
        
        lienzo_distribucion = FigureCanvasTkAgg(figura_distribucion, master=pestana_distribucion)
        lienzo_distribucion.draw()
        lienzo_distribucion.get_tk_widget().pack(fill="both", expand=True)
        
        # Grafico 2: Correlacion de Pearson horizontal
        pestana_correlaciones = ttk.Frame(cuaderno_graficos)
        cuaderno_graficos.add(pestana_correlaciones, text=" Correlacion de 11 Variables ")
        
        figura_correlacion, eje_correlacion = plt.subplots(figsize=(4.8, 3.2), dpi=92)
        variables_pearson = list(informacion_variables.keys())
        valores_pearson = [informacion_variables[v]['correlacion'] for v in variables_pearson]
        colores_barras = ['#28a745' if val >= 0 else '#dc3545' for val in valores_pearson]
        
        posiciones_y = np.arange(len(variables_pearson))
        eje_correlacion.barh(posiciones_y, valores_pearson, color=colores_barras, edgecolor='black', alpha=0.8)
        eje_correlacion.set_yticks(posiciones_y)
        eje_correlacion.set_yticklabels(variables_pearson, fontsize=7)
        eje_correlacion.axvline(0, color='black', linewidth=0.8, linestyle='--')
        eje_correlacion.set_xlabel("Coeficiente r de Pearson", fontsize=8)
        eje_correlacion.set_title("Impacto Estadistico sobre la Calidad", fontsize=9, fontweight='bold')
        eje_correlacion.grid(axis='x', linestyle='--', alpha=0.4)
        figura_correlacion.tight_layout()
        
        lienzo_correlacion = FigureCanvasTkAgg(figura_correlacion, master=pestana_correlaciones)
        lienzo_correlacion.draw()
        lienzo_correlacion.get_tk_widget().pack(fill="both", expand=True)

    # Fase 2: Induccion de reglas modulares PRISM y demostracion del Boundary Problem
    def construir_fase2(self):
        contenedor = ttk.Frame(self.pestana_fase2, padding=6)
        contenedor.pack(fill="both", expand=True)
        
        # Panel informativo del problema de fronteras rigidas
        marco_informacion = ttk.LabelFrame(contenedor, text=" Evaluacion de PRISM Rigido (Fronteras Booleanas) ", padding=6)
        marco_informacion.pack(fill="x", pady=(0, 6))
        
        exactitud_train_rigido = self.modelo.get('exactitud_entrenamiento_rigido', self.modelo.get('acc_train_rigido', 0))
        exactitud_test_rigido = self.modelo.get('exactitud_prueba_rigido', self.modelo.get('acc_test_rigido', 0))
        
        texto_problema_frontera = (
            f"• Total de Reglas Inducidas con PRISM: {len(self.reglas)} reglas modulares.\n"
            f"• Exactitud PRISM Rigido en Entrenamiento: {exactitud_train_rigido}%\n"
            f"• Exactitud PRISM Rigido en Prueba (Test) : {exactitud_test_rigido}%\n"
            f"(!) Problema de Frontera (Boundary Problem): Los modelos booleanos rigidos pierden precision (~33%)\n"
            f"    porque una pequena variacion quimica invalida la condicion discreta. La logica difusa resuelve esto en la Fase 3."
        )
        tk.Label(marco_informacion, text=texto_problema_frontera, font=("Segoe UI", 8), justify="left", bg="#fff3cd", fg="#856404", padx=8, pady=4).pack(fill="x")
        
        # Panel dividido: Tabla de reglas (Izquierda) y Analisis grafico (Derecha)
        marco_cuerpo = ttk.Frame(contenedor)
        marco_cuerpo.pack(fill="both", expand=True)
        
        # Izquierda: Tabla formal de reglas inducidas con Confianza, Soporte, Cobertura y Lift
        marco_tabla_reglas = ttk.LabelFrame(marco_cuerpo, text=" Base de Reglas Inducidas ", padding=4)
        marco_tabla_reglas.pack(side="left", fill="both", expand=True, padx=(0, 6))
        
        columnas_reglas = ("ID", "Antecedente (Condiciones)", "Consecuente", "Confianza", "Soporte", "Cobertura", "Lift")
        tabla_reglas = ttk.Treeview(marco_tabla_reglas, columns=columnas_reglas, show="headings")
        for columna in columnas_reglas:
            tabla_reglas.heading(columna, text=columna)
        tabla_reglas.column("ID", width=36, anchor="center")
        tabla_reglas.column("Antecedente (Condiciones)", width=300)
        tabla_reglas.column("Consecuente", width=75, anchor="center")
        tabla_reglas.column("Confianza", width=65, anchor="center")
        tabla_reglas.column("Soporte", width=60, anchor="center")
        tabla_reglas.column("Cobertura", width=65, anchor="center")
        tabla_reglas.column("Lift", width=55, anchor="center")
        
        barra_reglas = ttk.Scrollbar(marco_tabla_reglas, orient="vertical", command=tabla_reglas.yview)
        tabla_reglas.configure(yscrollcommand=barra_reglas.set)
        tabla_reglas.pack(side="left", fill="both", expand=True)
        barra_reglas.pack(side="right", fill="y")
        
        conteo_consecuente = {'ALTA': 0, 'MEDIA': 0, 'BAJA': 0}
        precisiones_por_clase = {'ALTA': [], 'MEDIA': [], 'BAJA': []}
        
        for regla in self.reglas:
            condiciones_texto = " Y ".join([f"{k}={v}" for k, v in regla['condiciones'].items()])
            confianza_valor = regla.get('confianza', regla.get('precision', 0))
            soporte_valor = regla.get('soporte', 0)
            cobertura_valor = regla.get('cobertura', 0)
            lift_valor = regla.get('lift', 1.0)
            
            tabla_reglas.insert("", "end", values=(
                regla['id'], f"SI {condiciones_texto}", regla['consecuente'],
                f"{confianza_valor}%", f"{soporte_valor}%", f"{cobertura_valor}%", lift_valor
            ))
            consecuente = regla['consecuente']
            if consecuente in conteo_consecuente:
                conteo_consecuente[consecuente] += 1
                precisiones_por_clase[consecuente].append(confianza_valor)
                
        # Derecha: Graficos de distribucion y precision de las reglas PRISM
        marco_grafico_prism = ttk.LabelFrame(marco_cuerpo, text=" Gráfico: Analisis de Reglas Inducidas ", padding=4)
        marco_grafico_prism.pack(side="right", fill="both", expand=True)
        
        figura_prism, (eje_prism_clases, eje_prism_precision) = plt.subplots(2, 1, figsize=(4.6, 4.4), dpi=92)
        
        clases_ordenadas = ['BAJA', 'MEDIA', 'ALTA']
        cantidades_reglas = [conteo_consecuente[c] for c in clases_ordenadas]
        barras_cantidades = eje_prism_clases.bar(clases_ordenadas, cantidades_reglas, color=['#dc3545', '#ffc107', '#28a745'], edgecolor='black', alpha=0.85)
        eje_prism_clases.set_ylabel("N° Reglas", fontsize=8)
        eje_prism_clases.set_title("Reglas Inducidas por Clase Consecuente", fontsize=9, fontweight='bold')
        eje_prism_clases.grid(axis='y', linestyle='--', alpha=0.4)
        for barra, valor in zip(barras_cantidades, cantidades_reglas):
            eje_prism_clases.text(barra.get_x() + barra.get_width()/2, valor + 0.6, f"{valor}", ha='center', fontsize=8, fontweight='bold')
            
        precision_promedio = [np.mean(precisiones_por_clase[c]) if precisiones_por_clase[c] else 0 for c in clases_ordenadas]
        barras_precision = eje_prism_precision.bar(clases_ordenadas, precision_promedio, color=['#c82333', '#e0a800', '#218838'], edgecolor='black', alpha=0.85)
        eje_prism_precision.set_ylabel("Precision Prom. (%)", fontsize=8)
        eje_prism_precision.set_ylim(0, 105)
        eje_prism_precision.set_title("Precision Promedio de Reglas por Clase", fontsize=9, fontweight='bold')
        eje_prism_precision.grid(axis='y', linestyle='--', alpha=0.4)
        for barra, valor in zip(barras_precision, precision_promedio):
            eje_prism_precision.text(barra.get_x() + barra.get_width()/2, valor + 2, f"{valor:.1f}%", ha='center', fontsize=8, fontweight='bold')
            
        figura_prism.tight_layout()
        lienzo_prism = FigureCanvasTkAgg(figura_prism, master=marco_grafico_prism)
        lienzo_prism.draw()
        lienzo_prism.get_tk_widget().pack(fill="both", expand=True)

    # Fase 3: Logica Difusa continua de Mamdani y superacion del Boundary Problem
    def construir_fase3(self):
        contenedor = ttk.Frame(self.pestana_fase3, padding=6)
        contenedor.pack(fill="both", expand=True)
        
        marco_informacion = ttk.LabelFrame(contenedor, text=" Impacto de la Logica Difusa de Mamdani ", padding=6)
        marco_informacion.pack(fill="x", pady=(0, 6))
        
        exactitud_train_inicial = self.modelo.get('exactitud_entrenamiento_inicial', self.modelo.get('acc_train_inicial', 0))
        exactitud_test_inicial = self.modelo.get('exactitud_prueba_inicial', self.modelo.get('acc_test_inicial', 0))
        exactitud_test_rigido = self.modelo.get('exactitud_prueba_rigido', self.modelo.get('acc_test_rigido', 0))
        ganancia_difusa = round(exactitud_test_inicial - exactitud_test_rigido, 2)
        
        texto_explicacion = (
            f"• Exactitud Difusa Inicial (Entrenamiento): {exactitud_train_inicial}%\n"
            f"• Exactitud Difusa Inicial (Prueba / Test): {exactitud_test_inicial}%\n"
            f"✓ Superacion del Boundary Problem: Con grados de pertenencia continuos mu en [0, 1] y operador T-norma MIN,\n"
            f"  la exactitud en datos de prueba sube de {exactitud_test_rigido}% a {exactitud_test_inicial}% (+{ganancia_difusa} puntos porcentuales)."
        )
        tk.Label(marco_informacion, text=texto_explicacion, font=("Segoe UI", 8), justify="left", bg="#d4edda", fg="#155724", padx=8, pady=4).pack(fill="x")
        
        # Selector de variables para visualizacion de funciones de pertenencia
        marco_controles = ttk.Frame(contenedor)
        marco_controles.pack(fill="x", pady=(0, 4))
        
        ttk.Label(marco_controles, text="Seleccionar visualizacion de funciones de pertenencia:", font=("Segoe UI", 8, "bold")).pack(side="left", padx=4)
        
        columnas_quimicas = list(self.datos.get('columnas_quimicas', self.datos.get('cols_quimicas', [])))
        opciones_visualizacion = ["4 Variables Principales (Grid 2x2)"] + columnas_quimicas
        self.selector_variable_difusa = ttk.Combobox(marco_controles, values=opciones_visualizacion, state="readonly", width=34)
        self.selector_variable_difusa.current(0)
        self.selector_variable_difusa.pack(side="left", padx=6)
        self.selector_variable_difusa.bind("<<ComboboxSelected>>", self.actualizar_grafico_difuso)
        
        # Lienzo embebido para funciones de pertenencia difusa
        self.marco_grafico_difuso = ttk.LabelFrame(contenedor, text=" Gráfico: Funciones de Pertenencia Difusa (Triangulares y Trapezoidales) ", padding=4)
        self.marco_grafico_difuso.pack(fill="both", expand=True)
        
        self.figura_difusa = plt.Figure(figsize=(9.2, 4.4), dpi=92)
        self.lienzo_difuso = FigureCanvasTkAgg(self.figura_difusa, master=self.marco_grafico_difuso)
        self.lienzo_difuso.get_tk_widget().pack(fill="both", expand=True)
        
        self.dibujar_funciones_difusas()

    def actualizar_grafico_difuso(self, evento=None):
        self.dibujar_funciones_difusas()

    def dibujar_funciones_difusas(self):
        self.figura_difusa.clear()
        seleccion = self.selector_variable_difusa.get()
        cortes_base = self.datos['cortes_base']
        
        if seleccion == "4 Variables Principales (Grid 2x2)":
            ejes = self.figura_difusa.subplots(2, 2)
            variables_principales = ['alcohol', 'acidez_volatil', 'sulfatos', 'acido_citrico']
            for eje, variable in zip(ejes.flatten(), variables_principales):
                cortes = cortes_base[variable]
                valores_x = np.linspace(cortes['minimo'], cortes['maximo'], 250)
                pertenencias_baja = [modelo.pertenencia_baja(x, cortes['corte_bajo'], cortes['punto_medio']) for x in valores_x]
                pertenencias_media = [modelo.pertenencia_media(x, cortes['corte_bajo'], cortes['punto_medio'], cortes['corte_alto']) for x in valores_x]
                pertenencias_alta = [modelo.pertenencia_alta(x, cortes['punto_medio'], cortes['corte_alto']) for x in valores_x]
                
                eje.plot(valores_x, pertenencias_baja, label='Bajo', color='#1f77b4', linewidth=1.8)
                eje.plot(valores_x, pertenencias_media, label='Medio', color='#ff7f0e', linewidth=1.8)
                eje.plot(valores_x, pertenencias_alta, label='Alto', color='#2ca02c', linewidth=1.8)
                eje.set_title(variable.replace('_', ' ').title(), fontsize=8, fontweight='bold')
                eje.set_ylabel("Pertenencia (mu)", fontsize=7)
                eje.grid(True, linestyle='--', alpha=0.35)
                if variable == 'alcohol':
                    eje.legend(fontsize=7, loc='center right')
        else:
            eje = self.figura_difusa.subplots(1, 1)
            variable = seleccion
            cortes = cortes_base[variable]
            cortes_opt = self.cortes_optimizados[variable]
            valores_x = np.linspace(cortes['minimo'], cortes['maximo'], 300)
            
            # Curvas base iniciales
            pertenencias_baja = [modelo.pertenencia_baja(x, cortes['corte_bajo'], cortes['punto_medio']) for x in valores_x]
            pertenencias_media = [modelo.pertenencia_media(x, cortes['corte_bajo'], cortes['punto_medio'], cortes['corte_alto']) for x in valores_x]
            pertenencias_alta = [modelo.pertenencia_alta(x, cortes['punto_medio'], cortes['corte_alto']) for x in valores_x]
            
            eje.plot(valores_x, pertenencias_baja, label='Bajo (Inicial)', color='#1f77b4', linestyle='--', linewidth=1.6)
            eje.plot(valores_x, pertenencias_media, label='Medio (Inicial)', color='#ff7f0e', linestyle='--', linewidth=1.6)
            eje.plot(valores_x, pertenencias_alta, label='Alto (Inicial)', color='#2ca02c', linestyle='--', linewidth=1.6)
            
            # Curvas calibradas por Algoritmo Genetico
            pertenencias_baja_opt = [modelo.pertenencia_baja(x, cortes_opt['corte_bajo'], cortes_opt['punto_medio']) for x in valores_x]
            pertenencias_media_opt = [modelo.pertenencia_media(x, cortes_opt['corte_bajo'], cortes_opt['punto_medio'], cortes_opt['corte_alto']) for x in valores_x]
            pertenencias_alta_opt = [modelo.pertenencia_alta(x, cortes_opt['punto_medio'], cortes_opt['corte_alto']) for x in valores_x]
            
            eje.plot(valores_x, pertenencias_baja_opt, label='Bajo (Calibrado AG)', color='#0d47a1', linewidth=2.2)
            eje.plot(valores_x, pertenencias_media_opt, label='Medio (Calibrado AG)', color='#e65100', linewidth=2.2)
            eje.plot(valores_x, pertenencias_alta_opt, label='Alto (Calibrado AG)', color='#1b5e20', linewidth=2.2)
            
            eje.set_title(f"Funciones de Pertenencia Difusa — {variable.replace('_', ' ').title()}", fontsize=10, fontweight='bold')
            eje.set_xlabel(f"Valor Fisicoquimico [{cortes['minimo']} a {cortes['maximo']}]", fontsize=8)
            eje.set_ylabel("Grado de Verdad (mu)", fontsize=8)
            eje.grid(True, linestyle='--', alpha=0.4)
            eje.legend(fontsize=8, loc='center right')
            
        self.figura_difusa.tight_layout()
        self.lienzo_difuso.draw()

    # Fase 4: Calibracion evolutiva de cortes y pesos con Algoritmo Genetico (Holland, 1975)
    def construir_fase4(self):
        contenedor = ttk.Frame(self.pestana_fase4, padding=6)
        contenedor.pack(fill="both", expand=True)
        
        marco_cuerpo = ttk.Frame(contenedor)
        marco_cuerpo.pack(fill="both", expand=True)
        
        # Columna izquierda: Resumen de hiperparametros del AG y tabla de cortes
        marco_izquierdo = ttk.Frame(marco_cuerpo)
        marco_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 6))
        
        marco_resumen_genetico = ttk.LabelFrame(marco_izquierdo, text=" Hiperparámetros y Resultados del Algoritmo Genético (Holland, 1975) ", padding=6)
        marco_resumen_genetico.pack(fill="x", pady=(0, 6))
        
        configuracion = self.modelo.get('configuracion_genetico', self.modelo.get('config_ag', {
            'tamano_poblacion': 25, 'numero_generaciones': 25,
            'metodo_seleccion': 'torneo', 'tamano_torneo': 3,
            'probabilidad_cruce': 0.85, 'probabilidad_mutacion': 0.20,
            'elite_count': 2
        }))
        
        exactitud_train_inicial = self.modelo.get('exactitud_entrenamiento_inicial', self.modelo.get('acc_train_inicial', 0))
        exactitud_train_optimizada = self.modelo.get('exactitud_entrenamiento_optimizada', self.modelo.get('acc_train_optimizada', 0))
        exactitud_test_inicial = self.modelo.get('exactitud_prueba_inicial', self.modelo.get('acc_test_inicial', 0))
        exactitud_test_optimizada = self.modelo.get('exactitud_prueba_optimizada', self.modelo.get('acc_test_optimizada', 0))
        
        texto_genetico = (
            f"• Población (N): {configuracion['tamano_poblacion']} | Generaciones: {configuracion['numero_generaciones']} | Selección: {configuracion['metodo_seleccion'].upper()} (k={configuracion.get('tamano_torneo', 3)})\n"
            f"• Probabilidad de Cruce (Pc): {configuracion['probabilidad_cruce']*100:.1f}% | Probabilidad de Mutación (Pm): {configuracion['probabilidad_mutacion']*100:.1f}%\n"
            f"• Elitismo (Elite Count): {configuracion['elite_count']} individuos preservados | Criterio: Generaciones\n"
            f"• Exactitud Entrenamiento: {exactitud_train_inicial}% ➔ {exactitud_train_optimizada}% | Prueba: {exactitud_test_inicial}% ➔ {exactitud_test_optimizada}%\n"
            f"• Reglas Activas Calibradas (w > 0.1): {self.modelo['reglas_activas']} de {self.modelo['total_reglas']} reglas"
        )
        tk.Label(marco_resumen_genetico, text=texto_genetico, font=("Segoe UI", 8, "bold"), justify="left", bg="#e8f4f8", fg="#0c5460", padx=8, pady=5).pack(fill="x")
        
        # Tabla de cortes calibrados
        marco_cortes = ttk.LabelFrame(marco_izquierdo, text=" Cortes Difusos Calibrados por el AG (11 Variables) ", padding=4)
        marco_cortes.pack(fill="both", expand=True)
        
        columnas_cortes = ("Variable", "Minimo", "Corte Bajo (a)", "Punto Medio (b)", "Corte Alto (c)", "Maximo")
        tabla_cortes = ttk.Treeview(marco_cortes, columns=columnas_cortes, show="headings", height=8)
        for columna in columnas_cortes:
            tabla_cortes.heading(columna, text=columna)
            tabla_cortes.column(columna, width=78, anchor="center")
        tabla_cortes.column("Variable", width=110, anchor="w")
        
        for variable, cortes in self.cortes_optimizados.items():
            tabla_cortes.insert("", "end", values=(variable, cortes['minimo'], cortes['corte_bajo'], cortes['punto_medio'], cortes['corte_alto'], cortes['maximo']))
        tabla_cortes.pack(fill="both", expand=True)
        
        # Columna derecha: Curva de convergencia de aptitud y comparativa de fases
        marco_graficos_genetico = ttk.LabelFrame(marco_cuerpo, text=" Gráficos: Evolucion del Fitness y Comparativa por Fases ", padding=4)
        marco_graficos_genetico.pack(side="right", fill="both", expand=True)
        
        figura_genetico, (eje_convergencia, eje_comparativa) = plt.subplots(2, 1, figsize=(4.8, 4.4), dpi=92)
        
        historial_aptitud = self.modelo.get('historial_aptitud', self.modelo.get('historial_fitness', []))
        eje_convergencia.plot(range(len(historial_aptitud)), historial_aptitud, marker='o', color='#d62728', linewidth=1.8, markersize=3.5, label='Mejor Fitness')
        eje_convergencia.set_title("Curva de Convergencia del Fitness (Generaciones)", fontsize=9, fontweight='bold')
        eje_convergencia.set_xlabel("Generacion", fontsize=8)
        eje_convergencia.set_ylabel("Exactitud (%)", fontsize=8)
        eje_convergencia.grid(True, linestyle='--', alpha=0.4)
        eje_convergencia.legend(fontsize=7)
        
        etiquetas_fases = ['F2: PRISM\nRigido', 'F3: Difuso\nInicial', 'F4: Difuso\n+ AG']
        exactitudes_fases = [
            self.modelo.get('exactitud_prueba_rigido', self.modelo.get('acc_test_rigido', 0)),
            self.modelo.get('exactitud_prueba_inicial', self.modelo.get('acc_test_inicial', 0)),
            self.modelo.get('exactitud_prueba_optimizada', self.modelo.get('acc_test_optimizada', 0))
        ]
        colores_fases = ['#6c757d', '#17a2b8', '#28a745']
        
        barras_fases = eje_comparativa.bar(etiquetas_fases, exactitudes_fases, color=colores_fases, edgecolor='black', alpha=0.85, width=0.55)
        eje_comparativa.set_ylabel("Exactitud Test (%)", fontsize=8)
        eje_comparativa.set_ylim(0, 85)
        eje_comparativa.set_title("Evolucion de Exactitud en Datos de Prueba", fontsize=9, fontweight='bold')
        eje_comparativa.grid(axis='y', linestyle='--', alpha=0.4)
        for barra, valor in zip(barras_fases, exactitudes_fases):
            eje_comparativa.text(barra.get_x() + barra.get_width()/2, valor + 2, f"{valor}%", ha='center', fontsize=8, fontweight='bold')
            
        figura_genetico.tight_layout()
        lienzo_genetico = FigureCanvasTkAgg(figura_genetico, master=marco_graficos_genetico)
        lienzo_genetico.draw()
        lienzo_genetico.get_tk_widget().pack(fill="both", expand=True)

    # Fase 5: Simulador interactivo en tiempo real con defuzzificacion continua por Centroide (CoG)
    def construir_fase5(self):
        contenedor = ttk.Frame(self.pestana_fase5, padding=6)
        contenedor.pack(fill="both", expand=True)
        
        # Panel izquierdo: Presets tipicos y 11 deslizadores fisicoquimicos
        panel_izquierdo = ttk.LabelFrame(contenedor, text=" Ajuste de Variables Fisicoquimicas del Vino ", padding=6)
        panel_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 6))
        
        marco_presets = ttk.Frame(panel_izquierdo)
        marco_presets.pack(fill="x", pady=(0, 4))
        
        ttk.Label(marco_presets, text="Perfiles Tipicos:", font=("Segoe UI", 8, "bold")).pack(side="left", padx=2)
        
        boton_preset_alto = tk.Button(marco_presets, text="Vino ALTO", bg="#28a745", fg="white", font=("Segoe UI", 7, "bold"),
                                      relief="flat", cursor="hand2", padx=5, command=self.cargar_preset_alto)
        boton_preset_alto.pack(side="left", padx=2)
        
        boton_preset_medio = tk.Button(marco_presets, text="Vino MEDIO", bg="#ffc107", fg="#212529", font=("Segoe UI", 7, "bold"),
                                       relief="flat", cursor="hand2", padx=5, command=self.cargar_preset_medio)
        boton_preset_medio.pack(side="left", padx=2)
        
        boton_preset_bajo = tk.Button(marco_presets, text="Vino BAJO", bg="#dc3545", fg="white", font=("Segoe UI", 7, "bold"),
                                      relief="flat", cursor="hand2", padx=5, command=self.cargar_preset_bajo)
        boton_preset_bajo.pack(side="left", padx=2)
        
        boton_preset_promedio = tk.Button(marco_presets, text="Promedio", bg="#6c757d", fg="white", font=("Segoe UI", 7, "bold"),
                                          relief="flat", cursor="hand2", padx=5, command=self.cargar_preset_promedio)
        boton_preset_promedio.pack(side="left", padx=2)
        
        # Deslizadores con scrollbar
        lienzo_deslizadores = tk.Canvas(panel_izquierdo, bg="#ffffff", highlightthickness=0)
        barra_deslizadores = ttk.Scrollbar(panel_izquierdo, orient="vertical", command=lienzo_deslizadores.yview)
        marco_deslizadores = ttk.Frame(lienzo_deslizadores)
        
        marco_deslizadores.bind("<Configure>", lambda e: lienzo_deslizadores.configure(scrollregion=lienzo_deslizadores.bbox("all")))
        lienzo_deslizadores.create_window((0, 0), window=marco_deslizadores, anchor="nw")
        lienzo_deslizadores.configure(yscrollcommand=barra_deslizadores.set)
        
        lienzo_deslizadores.pack(side="left", fill="both", expand=True)
        barra_deslizadores.pack(side="right", fill="y")
        
        self.variables_deslizador = {}
        self.etiquetas_deslizador = {}
        
        columnas_quimicas = self.datos.get('columnas_quimicas', self.datos.get('cols_quimicas', []))
        informacion_variables = self.datos.get('informacion_variables', self.datos.get('info_variables', {}))
        
        for variable in columnas_quimicas:
            info = informacion_variables[variable]
            valor_minimo = float(info['minimo'])
            valor_maximo = float(info['maximo'])
            valor_promedio = float(info['promedio'])
            
            fila_deslizador = ttk.Frame(marco_deslizadores, padding=2)
            fila_deslizador.pack(fill="x", expand=True, pady=1)
            
            encabezado_fila = ttk.Frame(fila_deslizador)
            encabezado_fila.pack(fill="x")
            ttk.Label(encabezado_fila, text=info['nombre_formal'], font=("Segoe UI", 8, "bold")).pack(side="left")
            
            etiqueta_valor = ttk.Label(encabezado_fila, text=f"{valor_promedio:.2f}", font=("Segoe UI", 8, "bold"), foreground=self.color_primario)
            etiqueta_valor.pack(side="right")
            self.etiquetas_deslizador[variable] = etiqueta_valor
            
            variable_tkinter = tk.DoubleVar(value=valor_promedio)
            self.variables_deslizador[variable] = variable_tkinter
            
            def callback_cambio(valor, etiqueta=etiqueta_valor):
                etiqueta.config(text=f"{float(valor):.2f}")
                self.ejecutar_simulacion()
                
            ttk.Scale(fila_deslizador, from_=valor_minimo, to=valor_maximo, variable=variable_tkinter, command=callback_cambio).pack(fill="x")
            
        boton_ejecutar = tk.Button(
            panel_izquierdo, text="🧪 EJECUTAR INFERENCIA Y DIBUJAR DEFUZZIFICACION",
            bg=self.color_primario, fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", pady=5, command=self.ejecutar_simulacion
        )
        boton_ejecutar.pack(fill="x", side="bottom", pady=(4, 0))
        
        # Panel derecho: Resultado de calidad y grafico continuo de Mamdani
        panel_derecho = ttk.Frame(contenedor)
        panel_derecho.pack(side="right", fill="both", expand=True)
        
        marco_resultado = ttk.LabelFrame(panel_derecho, text=" Resultado de la Inferencia Difusa Mamdani ", padding=6)
        marco_resultado.pack(fill="x", pady=(0, 6))
        
        fila_resultado = ttk.Frame(marco_resultado)
        fila_resultado.pack(fill="x")
        
        self.etiqueta_calidad = tk.Label(fila_resultado, text="CALIDAD ---", font=("Segoe UI", 15, "bold"), fg="#2c3e50")
        self.etiqueta_calidad.pack(side="left", padx=8)
        
        self.etiqueta_centroide = tk.Label(fila_resultado, text="Puntaje Centroide (CoG): ---", font=("Segoe UI", 10, "bold"), fg="#5c142c")
        self.etiqueta_centroide.pack(side="right", padx=8)
        
        self.etiqueta_probabilidades = ttk.Label(marco_resultado, text="Pertenencias Agregadas: Alta: 0.0% | Media: 0.0% | Baja: 0.0%", font=("Segoe UI", 8))
        self.etiqueta_probabilidades.pack(anchor="w", padx=8, pady=(2, 0))
        
        # Grafico continuo de defuzzificacion por Centroide de Gravedad (CoG)
        self.marco_grafico_defuz = ttk.LabelFrame(panel_derecho, text=" Gráfico: Defuzzificacion Continua con Centroide de Gravedad (CoG) ", padding=4)
        self.marco_grafico_defuz.pack(fill="both", expand=True)
        
        self.figura_defuz, self.eje_defuz = plt.subplots(figsize=(5.6, 3.4), dpi=92)
        self.lienzo_defuz = FigureCanvasTkAgg(self.figura_defuz, master=self.marco_grafico_defuz)
        self.lienzo_defuz.get_tk_widget().pack(fill="both", expand=True)
        
        # Inferencia inicial con valores promedio
        self.ejecutar_simulacion()

    def aplicar_valores_diccionario(self, diccionario_valores):
        for variable, valor in diccionario_valores.items():
            if variable in self.variables_deslizador:
                self.variables_deslizador[variable].set(valor)
                self.etiquetas_deslizador[variable].config(text=f"{float(valor):.2f}")
        self.ejecutar_simulacion()

    def cargar_preset_alto(self):
        # Perfil fisicoquimico caracteristico de un vino de alta calidad (>=7)
        self.aplicar_valores_diccionario({
            'acidez_fija': 8.8,
            'acidez_volatil': 0.35,
            'acido_citrico': 0.48,
            'azucar_residual': 2.4,
            'cloruros': 0.065,
            'dioxido_azufre_libre': 14.0,
            'dioxido_azufre_total': 34.0,
            'densidad': 0.9958,
            'ph': 3.30,
            'sulfatos': 0.82,
            'alcohol': 12.2
        })

    def cargar_preset_medio(self):
        # Perfil caracteristico de vino de calidad media (6)
        self.aplicar_valores_diccionario({
            'acidez_fija': 8.3,
            'acidez_volatil': 0.52,
            'acido_citrico': 0.27,
            'azucar_residual': 2.5,
            'cloruros': 0.080,
            'dioxido_azufre_libre': 15.0,
            'dioxido_azufre_total': 46.0,
            'densidad': 0.9967,
            'ph': 3.31,
            'sulfatos': 0.65,
            'alcohol': 10.4
        })

    def cargar_preset_bajo(self):
        # Perfil caracteristico de vino de baja calidad (<=5)
        self.aplicar_valores_diccionario({
            'acidez_fija': 7.6,
            'acidez_volatil': 0.82,
            'acido_citrico': 0.05,
            'azucar_residual': 2.6,
            'cloruros': 0.115,
            'dioxido_azufre_libre': 18.0,
            'dioxido_azufre_total': 75.0,
            'densidad': 0.9980,
            'ph': 3.42,
            'sulfatos': 0.48,
            'alcohol': 9.2
        })

    def cargar_preset_promedio(self):
        columnas_quimicas = self.datos.get('columnas_quimicas', self.datos.get('cols_quimicas', []))
        informacion_variables = self.datos.get('informacion_variables', self.datos.get('info_variables', {}))
        valores_promedio = {columna: float(informacion_variables[columna]['promedio']) for columna in columnas_quimicas}
        self.aplicar_valores_diccionario(valores_promedio)

    def ejecutar_simulacion(self):
        columnas_quimicas = self.datos.get('columnas_quimicas', self.datos.get('cols_quimicas', []))
        valores_actuales = {variable: float(self.variables_deslizador[variable].get()) for variable in columnas_quimicas}
        
        clase_predicha, probabilidades, reglas_activadas = modelo.evaluar_vino_difuso(
            valores_actuales, self.cortes_optimizados, self.reglas, self.pesos_optimizados
        )
        
        colores_clases = {'ALTA': "#28a745", 'MEDIA': "#b58105", 'BAJA': "#dc3545"}
        self.etiqueta_calidad.config(text=f"CALIDAD {clase_predicha}", fg=colores_clases.get(clase_predicha, "#2c3e50"))
        self.etiqueta_probabilidades.config(
            text=f"Pertenencias: Alta = {probabilidades['ALTA']*100:.1f}% | Media = {probabilidades['MEDIA']*100:.1f}% | Baja = {probabilidades['BAJA']*100:.1f}%"
        )
        
        # Calculo de defuzzificacion continua por Centroide de Gravedad (Mamdani CoG)
        self.eje_defuz.clear()
        escala_z = np.linspace(0, 10, 400)
        
        # Funciones de salida en el universo de calidad [0 a 10]
        pertenencia_baja_salida = np.clip(np.maximum(0, (5.0 - escala_z) / 2.5), 0, 1)
        pertenencia_baja_salida[escala_z <= 2.5] = 1.0
        pertenencia_media_salida = np.clip(np.minimum((escala_z - 3.5) / 2.5, (8.5 - escala_z) / 2.5), 0, 1)
        pertenencia_alta_salida = np.clip(np.maximum(0, (escala_z - 6.0) / 2.5), 0, 1)
        pertenencia_alta_salida[escala_z >= 8.5] = 1.0
        
        # Truncamiento Mamdani por operador MIN
        area_baja = np.minimum(pertenencia_baja_salida, probabilidades['BAJA'])
        area_media = np.minimum(pertenencia_media_salida, probabilidades['MEDIA'])
        area_alta = np.minimum(pertenencia_alta_salida, probabilidades['ALTA'])
        area_agregada = np.maximum(area_baja, np.maximum(area_media, area_alta))
        
        # Formula del Centroide: CoG = sum(z * mu(z)) / sum(mu(z))
        denominador_area = np.sum(area_agregada)
        centroide_gravedad = np.sum(escala_z * area_agregada) / denominador_area if denominador_area > 0 else 5.8
        self.etiqueta_centroide.config(text=f"Puntaje Centroide (CoG): {centroide_gravedad:.2f} / 10")
        
        # Curvas base del universo de salida
        self.eje_defuz.plot(escala_z, pertenencia_baja_salida, ':', color='#dc3545', alpha=0.45, label='Baja (<=5)')
        self.eje_defuz.plot(escala_z, pertenencia_media_salida, ':', color='#ffc107', alpha=0.45, label='Media (6)')
        self.eje_defuz.plot(escala_z, pertenencia_alta_salida, ':', color='#28a745', alpha=0.45, label='Alta (>=7)')
        
        # Relleno de las areas truncadas
        self.eje_defuz.fill_between(escala_z, 0, area_baja, color='#dc3545', alpha=0.3)
        self.eje_defuz.fill_between(escala_z, 0, area_media, color='#ffc107', alpha=0.3)
        self.eje_defuz.fill_between(escala_z, 0, area_alta, color='#28a745', alpha=0.3)
        
        # Curva de agregacion final y linea del centroide
        self.eje_defuz.plot(escala_z, area_agregada, color='#2c3e50', linewidth=2, label='Area Agregada (MAX)')
        self.eje_defuz.axvline(centroide_gravedad, color='#5c142c', linestyle='--', linewidth=2.5, label=f'Centroide (CoG) = {centroide_gravedad:.2f}')
        
        self.eje_defuz.set_title(f"Superficie Difusa Mamdani — Veredicto: {clase_predicha}", fontsize=9, fontweight='bold', color=colores_clases.get(clase_predicha, '#2c3e50'))
        self.eje_defuz.set_xlabel("Escala de Calidad del Vino [0 a 10]", fontsize=8)
        self.eje_defuz.set_ylabel("Grado de Verdad (mu)", fontsize=8)
        self.eje_defuz.set_ylim(-0.02, 1.05)
        self.eje_defuz.grid(True, linestyle='--', alpha=0.35)
        self.eje_defuz.legend(loc='upper right', fontsize=7)
        self.figura_defuz.tight_layout()
        self.lienzo_defuz.draw()

def iniciar_interfaz_fases(modelo_completo):
    ventana = tk.Tk()
    aplicacion = InterfazVinoFases(ventana, modelo_completo)
    ventana.mainloop()

iniciar_interfaz = iniciar_interfaz_fases

if __name__ == "__main__":
    import os
    ruta_csv = os.path.join(os.path.dirname(__file__), "winequality-red.csv")
    print("Entrenando pipeline secuencial en memoria...")
    modelo_entrenado = modelo.entrenar_modelo_completo(ruta_csv)
    iniciar_interfaz_fases(modelo_entrenado)
