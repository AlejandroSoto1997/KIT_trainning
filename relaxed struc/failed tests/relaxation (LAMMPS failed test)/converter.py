def convert_to_lammps_data(oxdna_conf_file, oxdna_top_file, lammps_data_file):
    # Read oxDNA configuration file
    with open(oxdna_conf_file, 'r') as f:
        oxdna_conf_data = f.readlines()

    # Read oxDNA topology file
    with open(oxdna_top_file, 'r') as f:
        oxdna_top_data = f.readlines()

    # Process oxDNA data and write LAMMPS data file
    with open(lammps_data_file, 'w') as f:
        # Write LAMMPS header
        f.write('LAMMPS data file\n\n')

        # Process oxDNA topology data
        # ... (your code to convert oxDNA topology to LAMMPS format)

        # Write atom information
        f.write('\nAtoms\n\n')
        for line in oxdna_conf_data[3:]:
            # ... (your code to convert oxDNA configuration to LAMMPS format)

               if __name__ == "__main__":
                   oxdna_conf_file = 'path/to/oxdna_conf.dat'
                   oxdna_top_file = 'path/to/oxdna_top.top'
                   lammps_data_file = 'path/to/lammps_data.data'

        #convert_to_lammps_data(oxdna_conf_file, oxdna_top_file, lammps_data_file)


# Ejemplo de uso
convert_to_lammps_data('output.dat', 'output.top', 'file.data')


