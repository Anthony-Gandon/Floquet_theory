# Floquet_theory
Extension of the Floquet approach to multiple commensurate and incommensurate frequencies

Parameters for the drives and the atomic system can be changed in the parameters.py file.
Hyperparameters such as the number of atomic states (2 or 3), the number of Floquet blocs to use in the MMFT computation, the number of drives (0,1,2) must be specified when running the scripts.

Basic examples are

Open the folder comparison_FL_QT in a shell:
 
 python -m 2 4 2 (Nb_atomic_states, Nb_Floquet_Blocs, Nb_drives)
 python -m 2 8 2 (Nb_atomic_states, Nb_Floquet_Blocs, Nb_drives)
 
 python -m 2 4 1 (Nb_atomic_states, Nb_Floquet_Blocs, Nb_drives)


