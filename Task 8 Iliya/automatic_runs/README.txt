Automatic Consecutive Runs in oxDNA
This example demonstrates the process of running consecutive simulations in oxDNA.

Input:

Linker L2 sequence: TCTATTCGCATGACATTCACCGTAAG
Sticky ends: GAGAGTCAAT
Initial Configuration:
The initial configuration file "output.dat" (named by ox view) is provided after pre-relaxation in oxview. The single linker is distributed along the box using the following relaxation conditions:

Temperature: 25C
Salt Concentration: 0.1
Thermostat: Langevin
Steps: 1e8-1e10
Enabled Forces per Base Pair (Automatic Option):
Forces per base pair are enabled automatically to ensure that no hydrogen bonds are lost.

Usage:

Adjust the number of steps per temperature range in the code.
Run the code in the terminal using the following command:

python3 run.py

This will create the first two folders in the lowest temperature folder. Manually provide the initial configuration file.
Then rerun the code, and the simulations will proceed automatically.

Author: Alejandro