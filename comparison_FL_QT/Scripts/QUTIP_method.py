import qutip as qt
import numpy as np
import scipy as sp
import sympy as sm
import time
import matplotlib.pyplot as plt
from qutip.qobj import Qobj, isket
from qutip.sesolve import sesolve
from qutip.propagator import propagator
from Scripts import parameters
	
class Qt():
	def __init__(self, Nb_atomic_states, Nb_floquet_blocks, Nb_drives):
		self.Nb_drives = Nb_drives
		self. Nb_floquet_blocks = Nb_floquet_blocks
		self.H0, self.list_perturbations, self.frequencies, self.w, self.eps0 = parameters.get_parameters(Nb_atomic_states, Nb_floquet_blocks, Nb_drives)
		#print(self.list_perturbations)

	def methodeQuTip(self, sort):
		self.Hfwo = sm.lambdify('eps0',self.H0, modules="numpy")
		self.H = [self.H0]

		T = 2*np.pi/self.w

		list_eps0 = np.arange(0, 8*self.w, 0.01*self.w)
		args_eps = {}

		f_energies_list = []
		for eps0 in list_eps0:
			args_eps['eps0']= eps0
			# progress_bar = int(eps0*self.w * 800)
			# if progress_bar%10==1:
			# 	print("%", progress_bar)
			#H0 = - eps0/2.0 * qt.sigmaz()
			H0 = Qobj(self.Hfwo(eps0))
			if self.Nb_drives==1:
				H = [H0, [Qobj(self.list_perturbations[0]), lambda t, args:np.cos(self.frequencies[0]*t)]]
			elif self.Nb_drives==2:
				H = [H0, [Qobj(self.list_perturbations[0]), lambda t, args:np.cos(self.frequencies[0]*t)],[Qobj(self.list_perturbations[1]), lambda t, args:np.cos(self.frequencies[1]*t)]]
			f_modes_0, f_energies = qt.floquet_modes(H, T, args_eps, sort=sort)
			f_energies_list.append(f_energies)

		llist_eps0 = np.transpose(np.matrix([list_eps0 for i in f_energies]))
		#plt.plot(llist_delta, f_energies_list )
		#plt.show()
		return llist_eps0,f_energies_list