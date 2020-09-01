import matplotlib.pyplot as plt
from Scripts import MMFT_method
from Scripts import QUTIP_method
import time
import sys
import numpy as np

def main1():
	#Compares Floquet and Qutip and plot eigen-energies

	#Parameters to modify

	Nb_atomic_states = int(sys.argv[1])
	Nb_floquet_blocks = int(sys.argv[2])
	Nb_drives = int(sys.argv[3])
	print("Nb_atomic_states = ",Nb_atomic_states)
	print("Nb_floquet_blocks = ",Nb_floquet_blocks)
	print("Nb_drives = ",Nb_drives)

	# Nb_atomic_states = 2
	# Nb_floquet_blocks = 4
	# Nb_drives = 2

	H_MMFT = MMFT_method.MMFT(Nb_atomic_states, Nb_floquet_blocks, Nb_drives)
	H_QUTIP = QUTIP_method.Qt(Nb_atomic_states, Nb_floquet_blocks, Nb_drives)

	time1 = time.time()
	H_MMFT.build_HF()
	time_inter = time.time()
	xm, ym = H_MMFT.methodeMMFT()
	time2 = time.time()
	xq, yq = H_QUTIP.methodeQuTip(sort=False)
	time3 = time.time()

	print("MMFT : build ="+str(time_inter-time1)+ "+ diag = "+str(time2-time_inter))
	print('QUTIP = ', time3-time2)

	plt.plot(xq, yq, 'o', label='QuTiP')
	plt.plot(xm, ym, '+', label='MMFT')
	plt.legend()
	plt.show()

	# abs_yq = np.abs(np.transpose(yq)[0])
	# abs_ym = np.abs(np.transpose(ym))
	#dif_y = abs_ym - abs_yq
	#print(dif_y.shape)
	#fig, ax = plt.subplots(3,1)
	# ax[0].plot(xq, yq, 'o', label='QuTiP')
	# ax[1].plot(xm, ym, '+', label='MMFT')
	# ax[2].plot(xq[:,1], np.transpose(dif_y))
	# fig.legend()
	# plt.savefig("C:/Users/antho/Documents/Mines/Stage Sherbrooke/Floquet_git/comparison_FL_QT/Pictures/diff.png")
	#dif_y_norm = [np.linalg.norm(vec) for vec in dif_y]
	# bins = np.arange(0, 1, 0.1) # fixed bin size

	# print(dif_y_norm)
	# print(len(dif_y_norm))
	# plt.hist(dif_y_norm, bins=bins)
	# #plt.plot(xq[:,1], np.transpose(dif_y))
	# plt.show()

#main()

def main2():
	#Two drives
	Nb_floquet_blocks = 7
	Nb_drives = 2
	Hamil = MMFT_method.MMFT(Nb_floquet_blocks, Nb_drives)
	time1 = time.time()
	Hamil.build_HF()
	time2 = time.time()
	print(time2-time1)
	xm, ym = Hamil.methodeMMFT()
	time3 = time.time()
	xq, yq = QUTIP_method.methodeQuTip(Nb_drives)
	plt.plot(xq, yq, 'o', label='QuTiP')
	plt.plot(xm, ym, '+', label='MMFT')
	plt.legend()
	plt.show()
	print('total = ', time3-time1)
#main2()

def main3():
	# Evaluates precision and duration of Qutip and MMFT for different number of floquet blocks


	#Parameters to modify

	Nb_atomic_states = 2
	Nb_floquet_blocks_range = range(2,6)
	Nb_drives = 2

	MMFT_times = []
	Qutip_times = []
	precision_first_component = []

	for Nb_floquet_blocks in Nb_floquet_blocks_range:
		print(Nb_floquet_blocks)
		H_MMFT = MMFT_method.MMFT(Nb_atomic_states, Nb_floquet_blocks, Nb_drives)
		H_QUTIP = QUTIP_method.Qt(Nb_atomic_states, Nb_floquet_blocks, Nb_drives)
		time1 = time.time()
		H_MMFT.build_HF()
		time_inter = time.time()
		xm, ym = H_MMFT.methodeMMFT(howtoreduce=5)
		time2 = time.time()
		xq, yq = H_QUTIP.methodeQuTip(sort=True)
		time3 = time.time()
		MMFT_times.append(time2-time1)
		Qutip_times.append(time3-time2)
		plt.plot(np.abs(ym), np.abs(yq)[:,1])
		print(np.linalg.norm(np.abs(ym)-np.abs(yq)[:,1]))
		precision_first_component.append(np.linalg.norm(np.abs(ym)-np.abs(yq)[:,1]))
	plt.show()
	fig, ax = plt.subplots(3,1)
	ax[0].plot(Nb_floquet_blocks_range, Qutip_times, 'o', label='QuTiP')
	ax[1].plot(Nb_floquet_blocks_range, MMFT_times, '+', label='MMFT')
	ax[2].plot(Nb_floquet_blocks_range, precision_first_component)
	fig.legend()
	plt.savefig("C:/Users/antho/Documents/Mines/Stage Sherbrooke/Floquet_git/comparison_FL_QT/Pictures/main3_3.png")


def find_Nb_Floquet_blocks():
	Hamil = MMFT_method.MMFT()
	total_list = []
	Nbs = np.arange(0,10,1)
	for Nb in Nbs:
		print(Nb)
		Hamil.update_Nb_floquet(Nb)
		Hamil.build_perturbations()

		Hamil.build_HF()
		f_energies_list = Hamil.find_Nb_floquet()

		total_list.append(f_energies_list)
	plt.ylim((-np.pi/Hamil.T1,np.pi/Hamil.T1))

	plt.plot(Nbs, total_list)
	plt.show()
#find_Nb_Floquet_blocks()

def test_decode():
	Hamil = MMFT()
	Hamil.build_HF()
	print(Hamil.Hf.shape)
	for i in range(Hamil.Hf.shape[0]):
		print(i, Hamil.decode(i))
#test_decode()

i = 1

if i==1:
	main1()
elif i==2:
	main2()
elif i==3:
	main3()