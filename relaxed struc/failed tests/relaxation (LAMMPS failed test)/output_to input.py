def generate_lammps_input(data_file):
    # Read data from the .lammps file
    with open(data_file, 'r') as f:
        data_content = f.read()

    # Create a LAMMPS input script based on the data
    lammps_input = f"""
# Example LAMMPS Input Script

# Set simulation box size
units real
dimension 3
boundary p p p
box 0 20 x 0 20 y 0 20 z

# Define atom types and masses
atom_style ellipsoid
{data_content}

# Define potential and simulation parameters
pair_style lj/cut 10.0
pair_coeff * * 1.0 1.0
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# Run simulation
timestep 1.0
run 1000
"""

    # Write the generated input script to a new file
    input_script_file = data_file.replace('.lammps', '_input.in')
    with open(input_script_file, 'w') as f:
        f.write(lammps_input)

    print(f"Generated LAMMPS input script: {input_script_file}")

# Example usage
generate_lammps_input('output.lammps')
