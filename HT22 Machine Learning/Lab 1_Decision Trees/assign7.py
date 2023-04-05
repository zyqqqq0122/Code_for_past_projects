import monkdata as m
import dtree as d
import random
import drawtree_qt5 as qt
import matplotlib.pyplot as plt
import numpy as np


def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]


# monk1train, monk1val = partition(m.monk1, 0.6)


def best_prune(dataset, fraction):
	train, val = partition(dataset, fraction)
	t_o = d.buildTree(train, m.attributes)
	while():
		E_old = d.check(t_o, val)
		t_new = []
		t_new = d.allPruned(t_o)
		Emax = -1
		for i in range(len(t_new)):
			E = d.check(t_new[i], val)
			if E > Emax:
				best_idx = i
				Emax = E

		E_new = Emax

		if E_new <= E_old:
			break
		else:
			t_o = t_new[best_idx]

	return t_o



def best_frac(dataset, testset, runtime):

	fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	E_mean = np.zeros(len(fraction))
	E_var = np.zeros(len(fraction))

	for i in range(len(fraction)):
		f = fraction[i]
		E = np.zeros(runtime)
		for k in range(runtime):
			t_best = best_prune(dataset, f)
			E[k] = 1 - d.check(t_best, testset)

		E_mean[i] = np.mean(E)
		E_var[i] = np.std(E)

	print(E_mean)
	print(E_var)


MONK1 = m.monk1
MONK3 = m.monk3
best_frac(MONK1, m.monk1test, 30)
best_frac(MONK3, m.monk3test, 30)










		








# qt.drawTree(t_best)
# t_ori = d.buildTree(monk1train, m.attributes)
# print(d.check(t_ori, monk1val))
# print(d.check(t_best, monk1val))

		





	
