#/Users/alejandrosoto/Documents/KIT/Task_7_Franc/auto_franc_appr
import os
import shutil
import subprocess

# Root directory where the folders for each temperature are located
root_dir = "/Users/alejandrosoto/Documents/KIT/Task_7_Franc/auto_franc_appr" # Root file with all the inputs 

# Path to the base configuration file
config_file_base_path = "/Users/alejandrosoto/Documents/KIT/Task_7_Franc/auto_franc_appr/input.txt" # Input file of reference
topology_file_path = os.path.join(root_dir, 'output.top')
#dist_file_path = os.path.join(root_dir, 'output.dat')
# Base configuration file format
config_file_base = """
## general input ###############
    backend = CPU #CUDA
    #backend_precision = mixed #uncomment when using CUDA
    fix_diffusion = 0
    back_in_box = 0
    use_edge = 1
    edge_n_forces = 1

## simulation options #############
    sim_type = MD
    ensemble = nvt
    dt = 0.005
    verlet_skin = 0.3
    steps = {}
    thermostat = langevin
    T = {}C
    newtonian_steps = 108
    diff_coeff = 2.5
    interaction_type = DNA2
    salt_concentration = 0.1
    max_backbone_force = 5.
    use_average_seq = no
    seq_dep_file = /Users/alejandrosoto/Documents/KIT/Tools/oxDNA/oxDNA/oxDNA2_sequence_dependent_parameters.txt #put your path here
    time_scale = linear

## input files ###########################################
    topology = output.top
    conf_file = output.dat

## output files ###########################################
    lastconf_file = last_conf_MD.dat
    trajectory_file = trajectory_MD.dat
    log_file = relax.output
    refresh_vel = 1
    no_stdout_energy = 0
    restart_step_counter = 1
    energy_file = energy_file_md.txt
    print_conf_interval = 100000.0
    print_energy_every = 100000.0

    data_output_1 = {{
        print_every = 1000.0
        name = hb_count.dat
        col_1 = {{
            type = hb_list
            only_count = True
        }}
    }}
    data_output_2 = {{
        print_every = 1000.0
        name = hb_list.dat
        col_1 = {{
            type = hb_list
        }}
    }}
"""

# List of temperatures to simulate
temperatures = [25, 30, 32, 35, 37, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# Number of steps per temperature range
steps_per_range = {
    (25, 40): 1000,
    (40, 70): 3000,
    (70, 100): 1000
}

# Iterate over each temperature
for i, temp in enumerate(temperatures):
    # Create directory for the current temperature
    temp_dir = os.path.join(root_dir, str(temp))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clean unnecessary files in the directory of the current temperature
    if i > 0:
        for filename in os.listdir(temp_dir):
            if filename != "input.txt" and filename != "output.top":
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    # Copy base configuration file to the directory of the current temperature
    shutil.copy(config_file_base_path, temp_dir)
    shutil.copy(topology_file_path, temp_dir)
    
    # If it's not the first temperature, copy the configuration file from the previous temperature
    if i > 0:
        prev_temp_dir = os.path.join(root_dir, str(temperatures[i - 1]))
        prev_config_file_path = os.path.join(prev_temp_dir, "last_conf_MD.dat")
        new_config_file_path = os.path.join(temp_dir, "output.dat")
        shutil.copy(prev_config_file_path, new_config_file_path)
        
    # Get the number of steps for this temperature according to the specified range
    steps = 1e3  # Default number of steps
    for temp_range, num_steps in steps_per_range.items():
        if temp_range[0] <= temp < temp_range[1]:
            steps = num_steps
            break
    
    # Generate content of the specific configuration file for this temperature
    config_file_content = config_file_base.format(steps, temp)
    
    # Write the configuration file to the directory of the current temperature
    with open(os.path.join(temp_dir, "input.txt"), "w") as f:
        f.write(config_file_content)
        
    # Run the simulation in oxDNA
    subprocess.run(["/Users/alejandrosoto/Documents/KIT/Tools/oxDNA/oxDNA-master/build/bin/oxDNA", "input.txt"], cwd=temp_dir)
