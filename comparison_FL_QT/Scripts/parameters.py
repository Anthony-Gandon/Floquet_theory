import numpy as np
import sympy as sm


def get_parameters(nb_atomic_states, nb_floquet_blocks, nb_drive):
	#1 Closed system parameters--- Energies of the different energy levels
	eps0 = sm.Symbol('eps0',real=True)

	delta = 0.2 * 2*np.pi

	if nb_atomic_states == 2:
		H0 = np.array([[-eps0, delta],[delta, eps0]])
	elif nb_atomic_states == 3:
		list_energies = [-eps0, 0, eps0]
		H0 = np.diagflat(list_energies)

	#2 Drive parameters

	#For 2 atomic states
	A = 0.5 * 2*np.pi
	B = 0.1 * 2*np.pi

	#For 3 atomic states
	Ka1=5
	Ka2=5
	Kb1=5
	Kb2=5

	omega1 = 10 * 2*np.pi
	omega2 = 25 * 2*np.pi
	omega  = 1 * 2*np.pi


	args = {'w1': omega1, 'w2':omega2}


	if nb_atomic_states == 2:
		drive1 = np.array([[0, A],[A, 0]])
		drive2 = np.array([[0, B],[B, 0]])
	if nb_atomic_states == 3:
		drive1 = np.array([[0, -Ka1, 0],[-Ka1,0,-Kb1],[0, -Kb1, 0]])
		drive2 = np.array([[0, -Ka2, 0],[-Ka2,0,-Kb2],[0, -Kb2, 0]])

	return H0, [drive1, drive2][:nb_drive], [omega1, omega2][:nb_drive], omega, eps0

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
	
def decode2(n, nb_drive, Nb_floquet_blocks, Nb_atomic_states):
	alpha = n%Nb_atomic_states
	N = n//Nb_atomic_states
	N_list = numberToBase(N,2*Nb_floquet_blocks+1, Nb_floquet_blocks, nb_drive)
	#print(N, N_list)
	#Works for Nb_floquet_blocks <= 16
	return alpha, N_list