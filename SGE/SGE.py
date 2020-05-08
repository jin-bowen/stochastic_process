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
	params1 = np.zeros((10, 4))
	params2 = np.zeros((10, 4))
	params1[:,0] = np.linspace(10, 100, 10)
	params1[:,1] = np.linspace(100, 10, 10)
	params1[:,2] = 100
	params1[:,3] = 1

	params2[:,0] = np.linspace(10, 100, 10)
	params2[:,1] = np.linspace(100, 10, 10)
	params2[:,2] = 10
	params2[:,3] = 1

	N = sys.argv[1]
	N = int(N)

	param = np.concatenate((params1, params2), axis=0)
	pop_0  = np.array([0,0])
	S = np.array([[1,-1,0,0],[0,0,1,-1]])
	sample_point = 100

	ensembl = np.zeros(( param.shape[0], N))
	for idx in range(param.shape[0]):
		params = param[idx,:]
		np.random.seed(3442)
		for i in range(N):
			pop, t = gillespie_algorithm(params, hazard, S, pop_0, T=10, sample_point=sample_point)
			ensembl[idx,i] = pop[-1,1]	
	np.savez('simulation_samp%s'%N, ensembl = ensembl, param = param)
	
#	ensembl = np.zeros(N)
#	fig, axs = plt.subplots(ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 1]} )
#	fig.suptitle('bursting expression pattern')
#	np.random.seed(3442)
#	for i in range(N):
#		pop, t = gillespie_algorithm(params, hazard, S, pop_0, T=10, sample_point=sample_point)
#		ensembl[i] = pop[-1,1]	
#		axs[0].plot(t, pop[:,1], 'b-', lw=0.5, alpha=0.5)
#	axs[0].grid()
#	axs[0].set_ylabel('mRNA copy number')
#	axs[0].set_xlabel('time')
#
#	axs[1].hist(ensembl, orientation='horizontal')
#	axs[1].grid()
#	axs[1].set_xlabel('number of cells')
#	plt.savefig('param2_SGE%s.png'%idx)

#	fig, axs = plt.subplots(nrows=3)
#	fig.suptitle('bursting expression pattern')
#	axs[0].set_title('kon=%s,koff=%s,ksync=%s'%(params[0], params[1], params[2]))
#	axs[0].plot(t, pop[:,0], '|', lw=1, alpha=0.6, label = 'active gene')
#	axs[0].set_ylim([0.9, 1.1])
#	axs[0].set_ylabel('promoter status')
#	axs[0].set_xlabel('time')
#	axs[0].set_yticks([])
#
#	axs[1].plot(t, pop[:,1], '-', lw=1, alpha=0.6)
#	axs[1].grid()
#	axs[1].set_ylabel('mRNA copy number')
#	axs[1].set_xlabel('time')
#
#	axs[2].hist(pop[:,1], density=True)
#	axs[2].axvline(x=np.mean(pop[:,1]), color='r')
#	axs[2].set_ylabel('probability')
#	axs[2].set_xlabel('mRNA copy number')
#
if __name__ == "__main__":
	main()

