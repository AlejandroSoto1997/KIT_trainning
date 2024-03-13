#### Automatic consecutive Runs in oxDNA ####

This example uses an input of the linker L2 with the sequence as double strand: TCTATTCGCATGACATTCACCGTAAG
And the sticky ends in both sides: GAGAGTCAAT

The box_size on this example is given but the initial configuration file "output.dat" (name given by ox view), the system was pre relaxed before in oxview in order to distribute the single linker along the box:

Condition of relaxation:

25°C
salt_concentration = 0.1
thermostat = langevin
steps = 1e8-1e10


ENABLED FORCES PER BASE PAIR (AUTOMATIC OPTION) IN ORDER DO NOT LOOSE ANY HYDROGEN BOND.

With this code you can change the number of steps per temperature range. 
