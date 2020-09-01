#test
import numpy as np
import sympy as sm
wo = sm.Symbol('wo',real=True)  #magentic dipole frequency
print(wo)
def decode1(n,Nb_floquet_blocks, No_subspaces = 0):
	# n = alpha+ (n1+2)*3+(n2+Nb_floquet_blocks)*(15)
	Nb_atomic_states = 3
	tot_size = Nb_atomic_states*(2*Nb_floquet_blocks+1)*(2*Nb_floquet_blocks+1)
	n-=No_subspaces*tot_size 
	alpha = n%Nb_atomic_states
	N = n//Nb_atomic_states
	n2 = N//(2*Nb_floquet_blocks+1)-Nb_floquet_blocks
	n1 = N%(2*Nb_floquet_blocks+1)-Nb_floquet_blocks
	return alpha, n1, n2

def numberToBase(n, b, Nb_floquet_blocks, nb_drive):
	if n == 0:
		return [-Nb_floquet_blocks]*nb_drive
	digits = []
	while n:
		digits.append(int(n % b))
		n //= b

	digits = np.pad(digits, (0, nb_drive-len(digits)), 'constant')
	#print(digits)
	digits-=np.array([Nb_floquet_blocks]*nb_drive)
	return digits
	
def decode2(n, nb_drive, Nb_floquet_blocks):
		# n = alpha+ (n1+2)*3+(n2+Nb_floquet_blocks)*(15)
		# tot_size = 3*(2*Nb_floquet_blocks+1)*(2*Nb_floquet_blocks+1)
		Nb_atomic_states = 3
		alpha = n%Nb_atomic_states
		decoded = [alpha]
		N = n//Nb_atomic_states
		N_list = numberToBase(N,2*Nb_floquet_blocks+1, Nb_floquet_blocks, nb_drive)
		#print(N, N_list)
		#Works for Nb_floquet_blocks <= 16
		return alpha, N_list

print(decode1(10, 5))
print(decode2(10, 2, 5))
