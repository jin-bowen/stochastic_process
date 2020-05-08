import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def hazard1(params, pop):

	kon, koff, s0, s1, d0, d1 = params
	gon, m, p = pop

	return np.array([ gon * koff,
			(2 - gon) * kon,
			gon  * s0,  
			m * d0,
			m * s1,
			p * d1])


def hazard(params, pop):

	kon, koff, s0, s1, d0, d1 = params
	gon, m, p = pop

	return np.array([ gon * koff,
			(2 - gon) * kon,
			gon  * s0,  
			m * d0,
			m * s1,
			p * d1])

#def kon_p(pop1, pop2, inter_params):
#
#	k0, k1 = 0.34, 2.15
#	gon1, m1, p1 = pop1
#	gon2, m2, p2 = pop2
#	m_ij, m_ii, s_ij, s_ii, theta_ii, theta_ij = inter_params	
#
#	phi  = np.exp(theta_ii)
#	phi *= 1 + np.exp(theta_ij) * (p2/s_ij)**(m_ij)	
#	phi /= 1 + (p2/s_ij)**(m_ij)
#
#	kon_new = k0 + k1 * phi * (p1/s_ii)**(m_ii)
#	kon_new /= 1 + phi * (p1/s_ii)**(m_ii)
#	return kon_new
 
def kon_p(pop1, pop2, kon2, ka=100):

	gon1, m1, p1 = pop1
	gon2, m2, p2 = pop2
	
	scale = (p1 + 1) /(ka + p1)
	kon_new = kon2 * scale

	return kon_new
 
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

def gillespie_algorithm(params1, params2, hazard1, hazard, S, T, pop_0, 
			sample_point=200):

	i = 0
	cum_t1 = 0
	cum_t2 = 0
	dt = float(T - 0.0)/float(sample_point)

	# starting population is the same for gene1 and gene2
	pop1 = pop_0.copy()
	pop2 = pop_0.copy()

	r_t   = np.linspace(0, T, num = sample_point + 1)
	r_pop1 = np.empty( (S.shape[0], sample_point + 1), dtype=np.int)
	r_pop2 = np.empty( (S.shape[0], sample_point + 1), dtype=np.int)
	r_pop1[:,0:] = pop1.reshape((-1,1))
	r_pop2[:,0:] = pop2.reshape((-1,1))

	while(cum_t1 <= T):
		# upstream gene
		j1, tau1 = gillespie_draw(params1, hazard1, pop1)
		cum_t1 += tau1
		i = int(np.floor(cum_t1/dt))

		pop1 += S[:,j1]
		r_pop1[:,i:] = pop1.reshape((-1,1))

		# downstream gene
		# kon, koff, s0, s1, d0, d1 = params
		# gon, m, p = pop
		# m_ij, m_ii, s_ij, s_ii, theta_ii, theta_ij = inter_params

		#update kon
		kon_new = kon_p(pop1, pop2, 0.6)
		params2[0] = kon_new

		while(cum_t2 <= cum_t1):
			j2, tau2 = gillespie_draw(params2, hazard, pop2)
			cum_t2 += tau2
			j = int(np.floor(cum_t2/dt))
			pop2 += S[:, j2]
			r_pop2[:,j:] = pop2.reshape((-1,1))

	return r_pop1, r_pop2, r_t

def main():

	# mRNA expression with two state model
	# kon, koff, s0, s1, d0, d1 = params
	params1 = np.array([ 0.6, 0.3, 2, 1, 10, 0.3 ])
	params2 = np.array([ 0.6, 0.3, 10, 1, 10, 0.03 ])

	T = 2000
	n_sim = 1
	pop_0  = np.array([0,0,0])
	S = np.array([[-2,2,0,0,0,0],
			[0,0,1,-1,0,0],
			[0,0,0,0,1,-1]])

	np.random.seed(42)
	pop1, pop2, t = gillespie_algorithm(params1, params2, hazard1, hazard, S, T, pop_0, sample_point=500)

	fig, axs = plt.subplots(nrows=3, sharex=True,constrained_layout=True )
	fig.suptitle('regulated gene expresison pattern\ncoordinated biallelic expression of TF gene')
	axs[0].plot(t, pop1[0,:] + 0.5, '.', lw=1, alpha=0.6, label = 'TF gene')
	axs[0].plot(t, pop2[0,:], '|', lw=1, alpha=1, label = 'downstream gene')
	axs[0].legend()

	axs[0].set_ylim([0.6, 3])
	axs[0].set_ylabel('promoter status')
	plt.setp(axs[0].get_yticklabels(), visible=False)
	axs[0].tick_params(axis='both', which='both', length=0)

	axs[1].plot(t, pop1[1,:], '-', lw=1, alpha=0.6)
	axs[1].plot(t, pop2[1,:], '-', lw=1, alpha=1)

	axs[1].grid()
	axs[1].set_ylabel('mRNA copy number')

	axs[2].plot(t, pop1[2,:], '-', lw=1, alpha=0.6)
	axs[2].plot(t, pop2[2,:], '-', lw=1, alpha=1)

	axs[2].grid()
	axs[2].set_ylabel('protein copy number')
	axs[2].set_xlabel('time/arbitrary unit')
#	plt.show()
	plt.savefig('reg_two_coord_copy.png')

#	fig, axs = plt.subplots(nrows=2)
#	scRNA = np.random.choice(pop[:,1], 200)
#	kon, koff, s0, s1, d0, d1 = params
#	lamda = s0 * kon/(kon + koff)
#	X = np.arange(0,20)
#	axs[0].hist(scRNA, bins=20, density=True)
#	axs[0].plot(X, st.poisson.pmf(X, lamda), '-')
#	plt.show()

if __name__ == "__main__":
	main()
