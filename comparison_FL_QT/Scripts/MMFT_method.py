#SEMICLASSICAL MANY-MODE FLOQUET THEORY

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

#Build the Floquet matrix
#Diagonalise the Floquet matrix for each value of a parameter

class MMFT(object):
	def __init__(self, Nb_atomic_states, Nb_floquet_blocks, Nb_drives):
		self.Nb_floquet_blocks = Nb_floquet_blocks
		self.Nb_subspaces=1
		self.tot_size = 2*(2*self.Nb_floquet_blocks+1)*(2*self.Nb_floquet_blocks+1)
		self.Nb_drives = Nb_drives
		#Atomic states 
		self.Nb_atomic_states = Nb_atomic_states
		# self.delta = 0.2 * 2*np.pi
		# self.eps0 = sm.Symbol('eps0',real=True)
		#Drives
		# self.w1 = 10 * 2*np.pi;	self.T1 = 2*np.pi/self.w1;
		# self.w2 = 25 * 2*np.pi;	self.T2 = 2*np.pi/self.w2;
		# self.w  = 5 * 2*np.pi;	self.T  = 2*np.pi/self.w;
		self.sz = sm.Matrix([[1,0],[0,-1]])
		self.sx = sm.Matrix([[0,1],[1,0]])
		self.id = sm.Matrix([[1,0],[0,1]])

		self.H0, self.list_perturbations, self.frequencies, self.w, self.eps0 = parameters.get_parameters(Nb_atomic_states, Nb_floquet_blocks, Nb_drives)
		#print(self.list_perturbations)
		self.build_perturbations()

	def decode(self, n, No_subspaces = 0):
		#Given the index of the states \ket{n}, returns alpha, n1, n2,... such that \ket{n} = \ket{alpha, n1, n2, ...}
		n-=No_subspaces*self.tot_size 
		alpha = n%self.Nb_atomic_states
		N = n//self.Nb_atomic_states
		n2 = N//(2*self.Nb_floquet_blocks+1)-self.Nb_floquet_blocks
		n1 = N%(2*self.Nb_floquet_blocks+1)-self.Nb_floquet_blocks
		return alpha, n1, n2

	def create_C(self, p, w, H_temp):
		#Creates a block matrix

		C = (H_temp+p*w*np.eye(H_temp.shape[0]))
		return C

	def compose_matrices(self, N,C,X,w):
		# Composes block matrices into a matrix
		future_matrix=[]
		for i in range(N):
			future_lign = []
			for j in range(N):
				if i==j:
					future_lign.append(C + ((N-1)//2-i)*np.eye(C.shape[0])*w)
				elif i==j-1:
					future_lign.append(X)
				elif i== j+1:
					future_lign.append(X)
				else:
					future_lign.append(np.zeros(C.shape))
			future_matrix.append(future_lign)
		return np.block(future_matrix)

	def build_HF(self):
		#Builds the Floquet Matrix 
		def test(p,H):
			p2=p
			listeNi = [int(np.real(self.H0[i,i]/self.w)) for i in range(self.H0.shape[0])]
			for Ni in listeNi:
				p2=p2%Ni
			return p2==0

		p=0
		list_HF = []
		for i in range(self.Nb_subspaces):
			if i!=0:
				p+=1
				while self.test(p,self.H0) and p<10:
					p+=1

			Cp = self.create_C(p, self.w, self.H0)

			HF_p = Cp
			for i, perturbation in enumerate(self.list_perturbations_to_use):
				HF_p = self.compose_matrices(2*self.Nb_floquet_blocks+1, HF_p, perturbation, self.frequencies[i])
			list_HF.append(HF_p)
		HF = sp.linalg.block_diag(*list_HF)
		HF = sm.Matrix(HF)
		HF.simplify()
		self.Hf = HF
		#print(HF)
		return HF

	def diagonalize_HF(HF, reorder=False):
		#Diagonalises the Floquet Matrix
		#print(HF)
		#Diagonalization of HF_0
		evals, evec = sp.linalg.eig(HF)
		return evals, evec

	def reduce_eigenvalues(self, evals):
		#applies periodicity rules on the eignevalues
		correction_list = []
		for n in range(len(evals)):
			alpha, n_list = parameters.decode2(n, self.Nb_drives, self.Nb_floquet_blocks, self.Nb_atomic_states)
			correction_list.append(-np.dot(n_list[:self.Nb_drives], np.array(self.frequencies)))
		corrected_list = np.sort(correction_list)[::-1]
		#print(' '.join(map(str, corrected_list)))  

		evals2 = np.array(evals)+np.array(corrected_list)
		evals2 = np.remainder(evals2+self.w/2,self.w)-self.w/2
		#print(len(np.diff(evals2)))
		#evals2[np.abs(np.diff(evals2, append=1)) >= 1] = np.nan

		return evals2
		#print("after", evals)

	def build_perturbations(self):
		# Converts the drive hamiltonians into the desired form
		new_list = []
		for i, perturbation in enumerate(self.list_perturbations):
			half_perturbation = perturbation/2
			new_list.append(sp.linalg.block_diag(*(half_perturbation,)*(2*self.Nb_floquet_blocks+1)**i))
		self.list_perturbations_to_use = new_list

	def update_Nb_floquet(self, Nb_floquet_blocks):
		self.Nb_floquet_blocks = Nb_floquet_blocks

	def methodeMMFT(self, howtoreduce=4):
		# Loops over a parameter and find the quasi-energies
		Hf_eps0 = sm.lambdify('eps0',self.Hf)
		list_eps0 = np.arange(0, 8*self.w, 0.01*self.w)
		size_eps0 = np.size(list_eps0)
		size_aut = self.Hf.shape[1]
		graph = np.empty([size_eps0,size_aut])
		#print(size_aut)

		f_energies_list = []
		for eps0 in list_eps0:
			# progress_bar = int(eps0/list_eps0[-1] * 800)
			# if progress_bar%10==0:
			# 	print("%", progress_bar)
			evals, evec = np.linalg.eig(Hf_eps0(eps0))
			evals = np.real(evals)
			evals = np.sort(evals)

			if howtoreduce==0:
				f_energies_list.append(evals)
			elif howtoreduce==1:
				#apply perdiodicity rule
				evals1 = self.reduce_eigenvalues(evals)[4:-4]
				f_energies_list.append(evals1)
			elif howtoreduce==2:
				 #Only keeps the smallest quasi-energy 
				evals2 = np.sort(np.abs(evals))[:1]
				f_energies_list.append(evals2)
			elif howtoreduce==3:
				evals3 = np.remainder(evals+self.w/2,self.w)-self.w/2
				f_energies_list.append(evals3)
			elif howtoreduce==4:
				evals3 = np.remainder(evals[1:-1]+self.w/2,self.w)-self.w/2
				f_energies_list.append(evals3)
			elif howtoreduce==5:
				evals2 = np.sort(np.abs(evals))[:1]
				evals3 = np.remainder(evals2+self.w/2,self.w)-self.w/2
				f_energies_list.append(evals3)


		if isinstance(f_energies_list[0],list):
			llist_eps0 = np.transpose(np.matrix([list_eps0 for i in range(len(f_energies_list[0]))]))
		else:
			llist_eps0 = list_eps0
		return llist_eps0, f_energies_list
		#plt.ylim((-np.pi/self.T1,np.pi/self.T1))
		#plt.plot(llist_eps0, f_energies_list )

	# def find_Nb_floquet(self):
	# 	Hf_eps0 = sm.lambdify('eps0',self.Hf)
	# 	list_eps0 = np.arange(0, 8*self.w, 1*self.w)
	# 	size_eps0 = np.size(list_eps0)
	# 	size_aut = self.Hf.shape[1]
	# 	graph = np.empty([size_eps0,size_aut])
	# 	f_energies_list = []
	# 	H1 = self.A/2.0 * qt.sigmax()
	# 	H2 = self.B/2.0 * qt.sigmax()

	# 	for eps in list_eps0:
	# 		evals, evec = np.linalg.eig(Hf_eps0(eps))
	# 		evals = np.real(evals)
	# 		#f_energies_list.append(np.sort(evals))

	# 		H0_qt = - eps/2.0 * qt.sigmaz()
	# 		H_qt = [H0_qt, [H1, lambda t, args:np.cos(self.w1*t)],[H2, lambda t, args:np.cos(self.w2*t)]]
	# 		args = {}
	# 		f_modes_0, f_energies = qt.floquet_modes(H_qt, self.T, args)
	# 		new_f_energies = [f_energies[0] for i in range(len(evals)//2)]
	# 		new_f_energies_end = [f_energies[1] for i in range(len(evals)//2)]
	# 		f_energies_2 = np.concatenate((new_f_energies, new_f_energies_end), axis=None)

	# 		f_energies_list.append(np.min(np.abs(np.sort(evals-f_energies_2))))
	# 	return np.mean(f_energies_list)





