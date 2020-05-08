import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def hazard(params, pop):

	kp, kn = params
	A,D = pop

	return np.array([ max(0.5*kp*(A-1)*A,0), max(kn*D,0)])

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
	r_pop[0:,:] = pop

	cum_t = 0
	while(cum_t < T):

		j, tau = gillespie_draw(params, hazard, pop)

		cum_t += tau
		i = int(np.ceil(cum_t/dt))
		pop += S[:,j]
		r_pop[i:,:] = pop


	return r_pop, r_t

def cle(params, hazard, S, pop_0, T, sample_point):

	omega = np.sum(pop_0) 
	pop = pop_0.copy() 	

	dt = float(T - 0.0)/float(sample_point)

	r_pop = np.zeros((sample_point + 1, pop.shape[0]))
	r_t   = np.linspace(0, T, num = sample_point + 1)
	r_pop[0,:] = pop

	z = st.norm.rvs(size=(sample_point + 1, pop.shape[0]), loc=0, scale=np.sqrt(dt))

	for i in range(1,sample_point + 1):
		hzd = hazard(params, pop)
		print(hzd)

		part1  = np.dot(S, hzd) * dt		
		part3  = np.dot(S, np.sqrt(hzd) * z[i,:])
	
		pop += part1 + part3
		r_pop[i,:] = pop

	return r_pop

def main():

	omega = 10000.0
	D = 5
	params = np.array([0.0001,0.05])
	pop_0  = np.array([omega - 2 * D, D])
	S = np.array([[-2,2],[1,-1]])

	np.random.seed(42)

	fig, ax = plt.subplots()
	fig.suptitle('question3 gillespie simulation')
	ax.set_title('system size=%s'%omega)
	
	T=100	
	sample_point = 1000
	dt = T/sample_point
	r_pop, r_t = gillespie_algorithm(params, hazard, S, pop_0, T, sample_point)
	r1_pop = cle(params, hazard, S, pop_0, T, sample_point)
	ax.plot(r_t, r_pop[:,0], '-', lw=1, label='particle A, gillespie simulation',)
	ax.plot(r_t, r1_pop[:,0], '-', lw=1, label='particle A, CLE')

	ax.set_xlabel('time points')
	ax.set_ylabel('number of particle')
	ax.grid(True)
	ax.legend()
	plt.show()
#	plt.savefig('system1_SP.png')

if __name__ == "__main__":
	main()

