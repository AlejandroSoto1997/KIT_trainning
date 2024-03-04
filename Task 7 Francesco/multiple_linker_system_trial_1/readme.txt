#### Trial 1 ####


The input files are refer to a multiple system with 20 linkers (L1) with the sequence:
L1-1: 5' - GAG AGT CAA TC TCT ATT CGC ATG ACA TTC ACC GTA AG - 3'
L1-2: 5' - GAG AGT CAA TC CTT ACG GTG AAT GTC ATG CGA ATA GA - 3'

The box size of the multiple system were calculated based on the box size of the single system (19 length units on oxDNA)

20*19=380

The target density of the final density should not be too different compared with the experimental values I think 1 g/cm3 to 1.25 g/cm3

The thermostat was changed to Langevin because many authors are using this to compare the experimental set up with the simulation at least for single systems I am not sure if this can be extended to multiple systems.

Also the code line of max_density_multiplier parameter was changed because the system is larger if the density is very low it can crush the simulation

Alejandro

alejandro.soto@kit.edu
