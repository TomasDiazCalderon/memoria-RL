import pandas as pd
import numpy as np

def procesar_comparativa(input_file="hullSF-cluster.csv", output_file="comparativa_resumen.csv"):
    # 1. Carga del dataset original
    df = pd.read_csv(input_file)

    # 2. Estandarización numérica (manejo de comas decimales y valores no disponibles/crashes)
    df['Tiempo_CPU'] = pd.to_numeric(df['Tiempo_CPU'].astype(str).str.replace(',', '.'), errors='coerce')
    df['Nodos'] = pd.to_numeric(df['Nodos'], errors='coerce')

    # 3. Cálculo de métricas estadísticas agregadas por Problema y Método
    stats = df.groupby(['Problema', 'Metodo']).agg(
        Corridas_Validas=('Tiempo_CPU', 'count'),
        Tiempo_Media=('Tiempo_CPU', 'mean'),
        Tiempo_Std=('Tiempo_CPU', 'std'),
        Tiempo_Min=('Tiempo_CPU', 'min'),
        Tiempo_Max=('Tiempo_CPU', 'max'),
        Nodos_Media=('Nodos', 'mean'),
        Nodos_Std=('Nodos', 'std'),
        Nodos_Min=('Nodos', 'min'),
        Nodos_Max=('Nodos', 'max')
    ).round(2)

    # 4. Formateo en columnas paralelas (lado a lado)
    pivot = stats.unstack(level='Metodo')
    pivot.columns = [f"{metodo}_{metrica}" for metrica, metodo in pivot.columns]
    resumen = pivot.reset_index()

    # 5. Métricas comparativas relativas (Tomando CLUSTER como referencia base)
    # Variación porcentual: ((HULL - CLUSTER) / CLUSTER) * 100
    resumen['Dif_Tiempo_%'] = (((resumen['HULL_Tiempo_Media'] - resumen['CLUSTER_Tiempo_Media']) / resumen['CLUSTER_Tiempo_Media']) * 100).round(2)
    resumen['Dif_Nodos_%'] = (((resumen['HULL_Nodos_Media'] - resumen['CLUSTER_Nodos_Media']) / resumen['CLUSTER_Nodos_Media']) * 100).round(2)
    resumen['Speedup_Hull'] = (resumen['CLUSTER_Tiempo_Media'] / resumen['HULL_Tiempo_Media']).round(2)

    # 6. Reordenamiento estructural de columnas
    columnas_ordenadas = [
        'Problema',
        # Tiempos CPU (s)
        'CLUSTER_Tiempo_Media', 'CLUSTER_Tiempo_Std', 'CLUSTER_Tiempo_Min', 'CLUSTER_Tiempo_Max',
        'HULL_Tiempo_Media', 'HULL_Tiempo_Std', 'HULL_Tiempo_Min', 'HULL_Tiempo_Max',
        'Dif_Tiempo_%', 'Speedup_Hull',
        # Nodos Explorados
        'CLUSTER_Nodos_Media', 'CLUSTER_Nodos_Std', 'CLUSTER_Nodos_Min', 'CLUSTER_Nodos_Max',
        'HULL_Nodos_Media', 'HULL_Nodos_Std', 'HULL_Nodos_Min', 'HULL_Nodos_Max',
        'Dif_Nodos_%',
        # Conteo de corridas completadas
        'CLUSTER_Corridas_Validas', 'HULL_Corridas_Validas'
    ]
    resumen = resumen[columnas_ordenadas]

    # 7. Exportación a archivo CSV
    resumen.to_csv(output_file, index=False)
    print(f"Archivo comparativo guardado como: {output_file}\n")
    return resumen

if __name__ == "__main__":
    df_resultado = procesar_comparativa()
    
    # Vista previa en consola de las métricas centrales
    columnas_vista = [
        'Problema', 
        'CLUSTER_Tiempo_Media', 'HULL_Tiempo_Media', 'Dif_Tiempo_%',
        'CLUSTER_Nodos_Media', 'HULL_Nodos_Media', 'Dif_Nodos_%'
    ]
    print(df_resultado[columnas_vista].to_string(index=False))