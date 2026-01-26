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

# 2. LISTA DE PROBLEMAS
PROBLEMS=(
    "test_infinity2.bch"
    "ex4_1_5.bch"
    "ex14_2_7.bch"
    "pressure-vessel.bch"
    "like.bch"
    "ex9_1_4.bch"
    "ex9_1_8.bch"
    "batch"
    "ex2_1_7"
    "ex6_1_2.bch"
    "dualc8"
    "polak5"
    "aircraftb"
    "ex4_1_8.bch"
    "hs108"
    "hs093"
    "hs117"
    "chem.bch"
    "ex2_1_9.bch"
    "ex2_1_8"
    "ex14_1_9.bch"
    "hs103"
    "like-1.bch"
    "ex2_1_4.bch"
    "exinfinity2.bch"
    "dualc1"
    "ex7_3_5bis.bch"
    "ex8_4inf-1.bch"
    "hs110"
    "ex8_1_8.bch"
    "ex6_2_11.bch"
    "chembis.bch"
    "ex2_1_2.bch"
    "hs093"
    "genhs28"
    "hs099"
    "ex14_1_2.bch"
    "ex8_1_5.bch"
    "ex8_1_6.bch"
    "disc2"
    "ex7_2_3bis.bch"
    "hs102"
    "ex8_4_4bis.bch"
    "test_infinity1.bch"
    "himmel16.bch"
    "ex9_2_5.bch"
    "ex8_1_2.bch"
    "ex6_2_8.bch"
    "ex7_2_5.bch"
    "dipigri.bch"
    "ex5_3_2.bch"
    "ex6_1_3.bch"
    "ex8_5_6-1.bch"
    "synthes1"
    "optmass"
    "hs056"
    "ex2_1_8"
    "hs091"
    "avgasb"
    "hs088"
    "ex8_5_4"
    "ex14_2_6"
    "ex8_5_1bis"
    "hs098"
    "makela4"
    "hs106"
    "makela3"
    "hs104"
    "ex9_2_2.bch"
    "ex2_1_7"

)

for vuelta in {1..5}; do
    echo "--- INICIANDO VUELTA $vuelta ---"
    
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

        echo "--- Ejecutando $problem_name (Vuelta $vuelta) ---"
        
        # 2. Ejecutar con timeout enviando SIGTERM al final
        timeout -s SIGTERM 3600s $IBEX_EXEC "$PROBLEM_PATH" &
        
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

    # Backup por vuelta
    cp "$Q_TABLE_FILE" "q_table_vuelta_$vuelta.bak"
done

echo "✅ ENTRENAMIENTO FINALIZADO."