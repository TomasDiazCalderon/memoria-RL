#!/bin/bash

# Función para limpiar y salir
cleanup() {
    echo -e "\n[!] Interrupción detectada. Matando procesos de ibex y saliendo..."
    # Mata a todos los procesos hijos del script
    pkill -P $$ 
    exit 1
}

# Registrar la función para CTRL+C (SIGINT) y SIGTERM
trap cleanup SIGINT SIGTERM

# 1. CONFIGURACIÓN
IBEX_EXEC="./ibexopt"
BENCHS_ROOT="../../benchs/optim"
Q_TABLE_FILE="q_table_trained.txt"

# 2. LISTA DE PROBLEMAS
PROBLEMS=(
    "test_infinity2.bch"
    "ex4_1_8.bch"
    "ex2_1_4.bch"
    "exinfinity2.bch"
    "ex2_1_2.bch"
    "makela4"
    "ex9_2_2.bch"
    "pressure-vessel.bch"
    "ex9_1_4.bch"
    "ex9_1_8.bch"
    "ex14_1_9.bch"
    "ex8_1_6.bch"
    "ex9_2_5.bch"
    "ex8_1_2.bch"
    "hs098"
    "ex7_2_5.bch"
    "ex6_1_2.bch"
    "ex8_1_8.bch"
    "ex8_1_5.bch"
    "ex4_1_5.bch"
    "polak5"
    "synthes1"
    "ex8_4inf-1.bch"
    "test_infinity1.bch"
    "ex14_1_2.bch"
    "hs106"
    "makela3"
    "ex14_2_6"
    "ex7_2_3bis.bch"
    "ex5_3_2.bch"
    "genhs28"
    "avgasb"
    "ex8_5_4"
    "ex8_5_6-1.bch"
    "hs056"
    "aircraftb"
    "ex7_3_5bis.bch"
    "like-1.bch"
    "disc2"
    "like.bch"
    "dualc1"
    "dipigri.bch"
    "hs110"
    "hs104"
    "ex8_5_1bis"
    "ex2_1_9.bch"
    "ex2_1_8"
    "ex2_1_7"
    "himmel16.bch"
    "ex6_2_11.bch"
    "ex6_2_8.bch"
    "ex14_2_7.bch"
    "hs099"
    "ex6_1_3.bch"
    "ex8_4_4bis.bch"
    "hs088"
    "hs117"
    "hs108"
    "dualc8"
    "hs093"
    "batch"
    "hs103"
    "chem.bch"
    "optmass"
)

# Lista de semillas
SEEDS=(7 42 500 999 1337)


for seed in "${SEEDS[@]}"; do
    
    for problem_name in "${PROBLEMS[@]}"; do
        # Buscar el archivo del problema
        PROBLEM_PATH=$(find "$BENCHS_ROOT" -name "$problem_name" -print -quit)

        if [ -z "$PROBLEM_PATH" ]; then
            continue
        fi

        # 1. Obtener la fecha de modificación antes de la ejecución
        if [ -f "$Q_TABLE_FILE" ]; then
            last_mod_before=$(stat -c %Y "$Q_TABLE_FILE")
        else
            last_mod_before=0
        fi

        echo "--- Ejecutando $problem_name | Semilla: $seed ---"

        # Exportar variables para que el C++ las lea
        export IBEX_PROB_NAME="$problem_name"
        export IBEX_SEED="$seed"
        
        # 2. Ejecutar con timeout enviando SIGTERM al final
        timeout -s SIGTERM 910s $IBEX_EXEC "$PROBLEM_PATH" --random-seed=$seed --timeout=900 &
        
        wait $!
        exit_code=$?

        # 3. Verificar resultados
        if [ $exit_code -eq 124 ] || [ $exit_code -eq 15 ] || [ $exit_code -eq 143 ]; then
            echo "!!!!! TIMEOUT alcanzado en $problem_name. !!!!!!!!!!!"
            
            # Verificar si el archivo se actualizó en el último minuto
            if [ -f "$Q_TABLE_FILE" ]; then
                last_mod_after=$(stat -c %Y "$Q_TABLE_FILE")
                if [ "$last_mod_after" -gt "$last_mod_before" ]; then
                    echo "CONFIRMADO: El Signal Handler guardó la Tabla Q exitosamente."
                else
                    echo "ERROR: La Tabla Q no se actualizó. Revisa la frecuencia de timer.check()."
                fi
            fi
        elif [ $exit_code -eq 130 ] || [ $exit_code -eq 143 ]; then
            exit 1
        elif [ $exit_code -eq 0 ]; then
            echo "Finalizado con éxito: $problem_name"
        fi
    done

done

echo "ENTRENAMIENTO FINALIZADO."