#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Resumen Estadístico Comparativo a CSV
Entrada:  comparacion_todo.csv
Salida:   resumen_estadistico_comparativo.csv
"""

import pandas as pd
import numpy as np

def procesar_resumen_estadistico(archivo_entrada="comparacion_todo.csv", 
                                 archivo_salida="resumen_estadistico_comparativo.csv"):
    df = pd.read_csv(archivo_entrada)

    # 1. Limpieza de columnas de clustering
    cols_clustering = ['Reinicios_Gatillados', 'Clusteres_Formados', 'Hulls_Creados', 'Nodos_Unidos']
    for col in cols_clustering:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2. Manejo riguroso de Timeouts (>= 3600s o sin solución en Base)
    df['Es_Timeout'] = (df['Tiempo_CPU'] >= 3600.0) | (df['Valor_Obj'].isna())
    df['Tiempo_Calculo'] = df['Tiempo_CPU'].copy()
    df.loc[df['Es_Timeout'], 'Tiempo_Calculo'] = 3600.0

    # 3. Métricas de referencia (IBEX Base)
    df_base = df[df['Metodo'] == 'IBEX'].groupby('Problema').agg(
        Base_T_Mean=('Tiempo_Calculo', 'mean'),
        Base_T_Median=('Tiempo_Calculo', 'median'),
        Base_N_Mean=('Nodos', 'mean'),
        Base_N_Median=('Nodos', 'median')
    ).reset_index()

    # 4. Agregación estadística por Problema y Método
    summary = df.groupby(['Problema', 'Metodo']).agg(
        Muestras=('Semilla', 'count'),
        Timeouts=('Es_Timeout', 'sum'),
        Tiempo_Medio_s=('Tiempo_Calculo', 'mean'),
        Tiempo_Mediana_s=('Tiempo_Calculo', 'median'),
        Tiempo_Std_s=('Tiempo_Calculo', 'std'),
        Nodos_Medio=('Nodos', 'mean'),
        Nodos_Mediana=('Nodos', 'median'),
        Nodos_Std=('Nodos', 'std'),
        Valor_Obj_Medio=('Valor_Obj', 'mean'),
        Valor_Obj_Min=('Valor_Obj', 'min'),
        Valor_Obj_Max=('Valor_Obj', 'max'),
        Reinicios_Medio=('Reinicios_Gatillados', 'mean'),
        Clusteres_Medio=('Clusteres_Formados', 'mean'),
        Hulls_Medio=('Hulls_Creados', 'mean'),
        Nodos_Unidos_Medio=('Nodos_Unidos', 'mean')
    ).reset_index()

    # 5. Cálculo de métricas relativas frente a IBEX Base
    summary = pd.merge(summary, df_base, on='Problema')
    summary['Speedup_Tiempo_Medio_%'] = ((summary['Base_T_Mean'] - summary['Tiempo_Medio_s']) / summary['Base_T_Mean']) * 100.0
    summary['Speedup_Tiempo_Mediana_%'] = ((summary['Base_T_Median'] - summary['Tiempo_Mediana_s']) / summary['Base_T_Median']) * 100.0
    summary['Reduccion_Nodos_Media_%'] = ((summary['Base_N_Mean'] - summary['Nodos_Medio']) / summary['Base_N_Mean']) * 100.0
    summary['Reduccion_Nodos_Mediana_%'] = ((summary['Base_N_Median'] - summary['Nodos_Mediana']) / summary['Base_N_Median']) * 100.0

    # Eliminar columnas auxiliares de Base
    summary = summary.drop(columns=['Base_T_Mean', 'Base_T_Median', 'Base_N_Mean', 'Base_N_Median'])

    # Redondeo limpio
    float_cols = summary.select_dtypes(include=['float64']).columns
    summary[float_cols] = summary[float_cols].round(4)

    # Orden canónico de presentación
    orden_metodos = {'IBEX': 0, 'CLUSTERING': 1, 'ADAPTATIVO': 2, 'REINSERCION': 3}
    summary['metodo_rank'] = summary['Metodo'].map(orden_metodos)
    summary = summary.sort_values(by=['Problema', 'metodo_rank']).drop(columns=['metodo_rank'])

    # Ordenar columnas lógicamente
    cols_order = [
        'Problema', 'Metodo', 'Muestras', 'Timeouts',
        'Tiempo_Medio_s', 'Tiempo_Mediana_s', 'Tiempo_Std_s', 'Speedup_Tiempo_Medio_%', 'Speedup_Tiempo_Mediana_%',
        'Nodos_Medio', 'Nodos_Mediana', 'Nodos_Std', 'Reduccion_Nodos_Media_%', 'Reduccion_Nodos_Mediana_%',
        'Valor_Obj_Medio', 'Valor_Obj_Min', 'Valor_Obj_Max',
        'Reinicios_Medio', 'Clusteres_Medio', 'Hulls_Medio', 'Nodos_Unidos_Medio'
    ]
    summary = summary[cols_order]

    summary.to_csv(archivo_salida, index=False)
    print(f"[OK] Archivo generado exitosamente: {archivo_salida}")

if __name__ == "__main__":
    procesar_resumen_estadistico()