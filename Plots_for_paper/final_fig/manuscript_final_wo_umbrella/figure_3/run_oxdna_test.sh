#!/bin/bash

start_temp=22
end_temp=85
step=2

# Directorio base donde están las simulaciones
base_dir="/home/alejandrosoto/Documents/KIT/GPU_dir_sam/EXPLICIT_COUNTING/L_3/SALT_0.1"

# Ruta al archivo de topología original
topology_file="${base_dir}/TEMP_${start_temp}/RUN_0/topol.top"

# Verifica que el archivo de topología exista
if [ ! -f "$topology_file" ]; then
    echo "Error: Topology file $topology_file not found. Exiting."
    exit 1
fi

for ((temp=$start_temp; temp<=$end_temp; temp+=$step)); do
    echo "Processing temperature: $temp"
    
    current_temp_dir="${base_dir}/TEMP_${temp}/RUN_0"
    next_temp=$((temp + step))
    next_temp_dir="${base_dir}/TEMP_${next_temp}/RUN_0"
    
    echo "Current directory: $current_temp_dir"
    echo "Next directory: $next_temp_dir"
    
    # Verifica si el directorio actual existe
    if [ ! -d "$current_temp_dir" ]; then
        echo "Error: $current_temp_dir does not exist. Exiting."
        exit 1
    fi
    
    # Si es la primera temperatura, corre oxDNA usando restart.dat
    if [ $temp -eq $start_temp ]; then
        echo "Running oxdna for the first temperature: $temp"
        cd $current_temp_dir
        nohup /home/alejandrosoto/Documents/oxDNA/build/bin/oxDNA input_MD.dat > log &
        oxdna_pid=$!
        wait $oxdna_pid
        
        # Espera pasiva hasta que last_conf_MD.dat sea generado
        while [ ! -f "$current_temp_dir/last_conf_MD.dat" ]; do
            echo "Waiting for last_conf_MD.dat to be generated..."
            sleep 600  # Espera 1 minuto antes de verificar de nuevo
        done
    fi
    
    # Para temperaturas subsecuentes, verifica si last_conf_MD.dat existe
    if [ ! -f "$current_temp_dir/last_conf_MD.dat" ]; then
        echo "Error: $current_temp_dir/last_conf_MD.dat not found. Exiting."
        exit 1
    fi
    
    # Copia last_conf_MD.dat al directorio de la siguiente temperatura
    mkdir -p $next_temp_dir
    cp $current_temp_dir/last_conf_MD.dat $next_temp_dir/restart.dat
    
    # Copia el archivo de topología al directorio de la siguiente temperatura
    cp $topology_file $next_temp_dir/topol.top
    
    # Actualiza input_MD.dat para la siguiente temperatura
    cp $current_temp_dir/input_MD.dat $next_temp_dir/input_MD.dat
    sed -i "s/^T = .*/T = ${next_temp} C/" $next_temp_dir/input_MD.dat
    sed -i "s/^steps = .*/steps = 1e9/" $next_temp_dir/input_MD.dat
    
    # Depuración: imprime el contenido de input_MD.dat
    echo "Content of $next_temp_dir/input_MD.dat:"
    cat $next_temp_dir/input_MD.dat
    
    # Corre oxdna para la siguiente temperatura
    echo "Running oxdna for the temperature: $next_temp"
    cd $next_temp_dir
    nohup /home/alejandrosoto/Documents/oxDNA/build/bin/oxDNA input_MD.dat > log &
    oxdna_pid=$!
    wait $oxdna_pid
    
    # Espera pasiva hasta que last_conf_MD.dat sea generado
    while [ ! -f "$next_temp_dir/last_conf_MD.dat" ]; do
        echo "Waiting for last_conf_MD.dat to be generated..."
        sleep 600  # Espera 1 minuto antes de verificar de nuevo
    done
done
