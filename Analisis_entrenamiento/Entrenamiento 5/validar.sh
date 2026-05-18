#!/bin/bash

# --- CONFIGURACIÓN ---
EXEC="./ibexopt" 
BENCHS_ROOT="../../benchs/optim"
OUTPUT_FILE="resultados_finales_limpios_E5_2.csv"

# 1. SEGURIDAD: Respaldar archivos de entrenamiento originales
echo "[INFO] Respaldando archivos de entrenamiento..."
cp tau_state.txt tau_state.txt.bak 2>/dev/null
cp q_table_trained.txt q_table_trained.txt.bak 2>/dev/null

# 2. FORZAR EXPLOTACIÓN: Escribir 0.0 en el archivo que lee tu C++
echo "0.0" > tau_state.txt

# 3. PROTECCIÓN: Hacer la tabla Q de "solo lectura" para que el programa 
# no la modifique durante la validación (evita aprendizaje accidental)
chmod 444 q_table_trained.txt

# Semillas y Problemas
SEEDS=(7 15 252 500 850 999 1234 1337 1500 1723)
PROBLEMS=(
    "avion2"
)

echo "Problema,Semilla,Tiempo_CPU,Nodos,Estatus" > $OUTPUT_FILE

for prob in "${PROBLEMS[@]}"; do
    PATH_PROB=$(find "$BENCHS_ROOT" -name "$prob" -print -quit)
    if [ -z "$PATH_PROB" ]; then continue; fi

    for seed in "${SEEDS[@]}"; do
        echo ">>> Validando $prob | Semilla $seed"
        
        # Ejecución normal (sin variables de entorno)
        output=$(timeout 3610s $EXEC "$PATH_PROB" --random-seed=$seed --timeout=3600 2>&1)
        
        time_cpu=$(echo "$output" | grep "cpu time used:" | awk '{print $4}' | sed 's/s//')
        nodes=$(echo "$output" | grep "number of cells:" | awk '{print $4}')
        status=$(echo "$output" | grep -E "successful|time limit|infeasible" | head -n 1 | xargs)

        echo "$prob,$seed,$time_cpu,$nodes,$status" >> $OUTPUT_FILE
    done
done

# 4. RESTAURAR: Devolver todo a su estado original
echo "[INFO] Restaurando archivos de entrenamiento..."
chmod 644 q_table_trained.txt
mv tau_state.txt.bak tau_state.txt
mv q_table_trained.txt.bak q_table_trained.txt

echo "Validación terminada. Resultados en $OUTPUT_FILE"