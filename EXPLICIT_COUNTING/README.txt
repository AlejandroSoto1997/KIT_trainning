## multiply.py ##

takes as input an oxDNA data file with 1 DNA linker and its topology and returns the configuration and topology of a suspension of N linkers. Box size must be set

Usage: python3 multipy.py file topology_file new_box_size N_filaments
("python3 multiply.py -h" or "python3 multiply.py" give the same information)

The linkers are set on a cubic lattice.

## lancia.sh ##

Simple bash script for a sweep of the temperature. Set to simulate between 14C and 92C.
Except at 14C, takes as input file the last configuration of the previous simulated temperature.
(Note: not advisable at high temperatures!)

## input_MD.dat ##

template input script for oxDNA on GPU. 
The restart configuration file is set as "restart.dat"; the checkpoint file is called "last_conf_MD.dat" and is printed every 1E7 steps. Should be printed also if oxDNA gets terminated.
Using the "brownian" thermostat

