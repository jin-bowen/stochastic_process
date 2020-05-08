from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy as sp
import numpy as np
import random
import math
import sys

def hazard(params, pop):

	r, gamma, u, d, V  = params
	m, p  = pop

	return np.array([ r * V,  
			m * gamma,
			m * u,
			p * d])

def sample_discrete_scipy(probs):
	
	return st.rv_discrete(values=(range(len(probs)), probs)).rvs()

def gillespie_draw(params, hazard, pop):

	# Compute propensities
	prob = hazard(params, pop)
	
	# Sum of propensities
	prob_sum = prob.sum()
	
	# Compute time
	tau = np.random.exponential(1.0 / prob_sum)
	
	# Compute discrete probabilities of each reaction
	rxn_probs = prob / prob_sum
	
	# Draw reaction from this distribution
	rxn = sample_discrete_scipy(rxn_probs)
	
	return rxn, tau

def gillespie_algorithm(params, hazard, S, pop_0, T, sample_point=1000):

	pop = pop_0.copy()
	dt = float(T - 0.0)/float(sample_point)

	r_pop = np.zeros((sample_point + 1, pop.shape[0]))
	r_t   = np.linspace(0, T, num = sample_point + 1)

	cum_t = 0
	while(cum_t < T):
		j, tau = gillespie_draw(params, hazard, pop)
		cum_t += tau
		i = int(np.floor(cum_t/dt))
		r_pop[i:,:] = pop
		pop += S[:,j]

	return r_pop[1:, :]

def poisson(s, k, total_t, sample_point=1000):

	points = np.zeros(sample_point)
	points[:] = 0

	dt = float(total_t - 0)/float(sample_point)
	cum_t = 0
	while(cum_t < total_t): 	
		p = random.random()
		interval = -math.log(1.0 - p)/s
		cum_t += interval

		nt = int(np.floor(min(cum_t,total_t)/dt))
		
		jump_size = np.random.exponential(scale=k) 

		points[nt:-1] += jump_size

	return points

def ode(s, k, d, V, T, N=1000):

	dt = float(T - 0)/float(N)
	xt = np.zeros(N)
	for i in range(1,N):
		xt[i] = xt[i-1] - d * xt[i-1] * dt + s/k * dt

	return xt

def levy(s, k, d, T, N=1000):

	xt = np.zeros(N)
	dt = float(T - 0)/float(N)

	ct = poisson(s, k, T, sample_point=N+1)
	dct = ct[1:] - ct[0:N]

	for i in range(1,N):
		xt[i] = xt[i-1] - d * xt[i-1] * dt + dct[i-1]

	return xt		

def main():

	dt = 0.1
	N = 2000
	T = dt * N
	s = k = 1
	d = 0.1
	V = 10
	t = np.linspace(0, T, num=N+1)

	xt_sde = levy(s, k, d, T, N)
	xt_ode = ode(s, k, d, V, T, N) 

	pop_0  = np.array([0,0])
	S = np.array([[1,-1,0,0],[0,0,1,-1]])
	params = np.array([s, 10, k, 0.1, V])

	xt_gillespie = gillespie_algorithm(params, hazard, S, pop_0, T, sample_point=N)
	xt_gillespie = xt_gillespie


	ax = plt.figure()
	plt.suptitle('comparison among levy-driven SDE, ODE and two-state model')
	plt.title('k+=inf, k-=0, r=1, gamma=10, u=1, d=0.1 ')
	plt.plot(t[1:], xt_sde, '-', label='levy limit',   lw=1)
	plt.plot(t[1:], xt_ode, '-', label='kurtz limit',  lw=1)
	plt.plot(t[1:], xt_gillespie[:,1], '-', label='two-state model',  lw=1)
	plt.legend()
	plt.grid()
	plt.xlabel('time(arbitrary unit)')
	plt.ylabel('protein concentration(copy number/volume)')
#	plt.show()
	ax.savefig('levy.png')

if __name__ == "__main__":
	main()






