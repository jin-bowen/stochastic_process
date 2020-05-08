import numpy as np
from matplotlib import gridspec
import scipy.stats as st
import matplotlib.pyplot as plt
import sys

def hazard(params, pop):

	kon, koff, r, gamma = params
	gon, m  = pop

	return np.array([ (1-gon) * kon,
			gon * koff,
			gon  * r,  
			m * gamma])

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

	return r_pop, r_t

def main():

	# mRNA expression with two state model
	N = sys.argv[1]
	N = int(N)

	pop_0  = np.array([0,0])
	S = np.array([[1,-1,0,0],[0,0,1,-1]])
	sample_point = 100
	params_list = [[0.6, 0.3, 10, 1],[0.3, 0.6, 10, 1], [6, 3, 10, 1],[3, 6, 10, 1]]
	color_list = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

	ii = 1
	for params, color in zip(params_list, color_list):
		ensembl = np.zeros(N)
		fig, axs = plt.subplots(ncols=2,  gridspec_kw={'width_ratios': [3, 1]} )
		fig.suptitle('bursting expression pattern')
		np.random.seed(3442)
	
		for i in range(1,N):
			pop, t = gillespie_algorithm(params, hazard, S, pop_0, T=100, sample_point=sample_point)
			ensembl[i] = pop[-1,1]	
			axs[0].plot(t, pop[:,1], '-', lw=0.3, c=color)
	
		pop, t = gillespie_algorithm(params, hazard, S, pop_0, T=100, sample_point=sample_point)
		axs[0].plot(t, pop[:,1], '-', c='k')
	
		ensembl = ensembl/ensembl.max()
		axs[0].grid()
		axs[0].set_title('k+=%s,k-=%s,r=%s,gamma=%s'%(params[0], params[1], params[2], params[3]))
		axs[0].set_ylabel('mRNA copy number')
		axs[0].set_xlabel('time')
	
		axs[1].hist(ensembl, orientation='horizontal', density=True, color=color)
		axs[1].set_ylim(0,1)
		axs[1].grid()
		axs[1].set_xlabel('number of cells')
		plt.savefig('param%s_SGE.png'%str(ii))
		ii += 1

if __name__ == "__main__":
	main()

