import numpy as np
from matplotlib import gridspec
import scipy.stats as st
import matplotlib.pyplot as plt
import sys

def hazard(params, pop):

	kon, koff, r, gamma, u, d  = params
	gon, m, p  = pop

	return np.array([ gon * koff,
			(1-gon) * kon,
			gon  * r,  
			m * gamma,
			m * u,
			p *d])

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

	pop_0  = np.array([0,0,0])
	S = np.array([[-1,1,0,0,0,0],[0,0,1,-1,0,0],[0,0,0,0,1,-1]])
	sample_point = 1000
	T = 1000
	params = [0.6, 0.3, 10, 1, 10, 0.03]
	color = '#1f77b4'	

#	params = [0.3, 0.6, 10, 1]
#	color = '#ff7f0e'	

#	params = [6, 3, 10, 1]
#	color = '#2ca02c'	


#	params = [3, 6, 10, 1]
#	color = '#d62728'	

	pop_ref, t = gillespie_algorithm(params, hazard, S, pop_0, T=T, sample_point=sample_point)
	pop_alt, t = gillespie_algorithm(params, hazard, S, pop_0, T=T, sample_point=sample_point)
	pop_all = pop_ref + pop_alt

	pop_alt_plot = -1 * pop_alt 

	fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, constrained_layout=True)
	fig.suptitle('bursting expression pattern\nk+=%s,k-=%s,r=%s, gamma=%s,u=%s, d=%s'%(params[0], params[1], params[2], params[3], params[4], params[5]))
	axs[0,0].plot(t, pop_ref[:,0], '-', lw=1, alpha=0.6, label = 'active gene')
	axs[0,0].plot(t, pop_alt_plot[:,0], '-', lw=1, alpha=0.6, label = 'active gene')
	axs[0,0].hlines(y=0, xmin=0, xmax=t[-1], colors='k', linestyles='-', lw=2)

	axs[0,0].set_ylim([-1.1, 1.1])
	axs[0,0].set_ylabel('promoter status')
	axs[0,0].set_xlabel('time')
	axs[0,0].set_yticks([])

	axs[1,0].plot(t, pop_ref[:,1], '-', lw=1, alpha=0.6)
	axs[1,0].plot(t, pop_alt_plot[:,1], '-', lw=1, alpha=0.6)
	axs[1,0].hlines(y=0, xmin=0, xmax=t[-1], colors='k', linestyles='-', lw=2)

	axs[1,0].grid()
	axs[1,0].set_ylabel('mRNA copy number')
	axs[1,0].set_xlabel('time')

	axs[2,0].plot(t, pop_ref[:,2], '-', lw=1, alpha=0.6)
	axs[2,0].plot(t, pop_alt_plot[:,2], '-', lw=1, alpha=0.6)
	axs[2,0].hlines(y=0, xmin=0, xmax=t[-1], colors='k', linestyles='-', lw=2)

	axs[2,0].grid()
	axs[2,0].set_ylabel('protein copy number')
	axs[2,0].set_xlabel('time')

	######
	axs[0,1].plot(t, pop_all[:,0], '-', lw=1, alpha=0.6, label = 'active gene')

	axs[0,1].set_ylim([0, 2.1])
	axs[0,1].set_ylabel('promoter status')
	axs[0,1].set_xlabel('time')
	axs[0,1].set_yticks([])

	axs[1,1].plot(t, pop_all[:,1], '-', lw=1, alpha=0.6)

	axs[1,1].grid()
	axs[1,1].set_ylabel('mRNA copy number')
	axs[1,1].set_xlabel('time')

	axs[2,1].plot(t, pop_all[:,2], '-', lw=1, alpha=0.6)

	axs[2,1].grid()
	axs[2,1].set_ylabel('protein copy number')
	axs[1,1].set_xlabel('time')

	plt.savefig('SGE_ind.png')

if __name__ == "__main__":
	main()

