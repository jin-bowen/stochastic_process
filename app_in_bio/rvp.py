import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def hazard(params, pop):

	beta, gamma = params
	S, I = pop
	N = S + I

	return np.array([ beta * S * I/N,
			gamma * I])

def sample_discrete_scipy(probs):
	
	return st.rv_discrete(values=(range(len(probs)), probs)).rvs()

def gillespie_draw(params, hazard, pop):

	# Compute propensities
	prob = hazard(params, pop)

	# Sum of propensities
	prob_sum = prob.sum()

	if prob_sum == 0: return [np.nan, np.nan]
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

	r_pop[:,0] = pop_0[0]
	r_pop[:,1] = pop_0[1]

	cum_t = 0
	while(cum_t < T):

		j, tau = gillespie_draw(params, hazard, pop)
		if np.isnan(tau): break
		cum_t += tau
		i = int(np.floor(cum_t/dt))

		pop += S[:,j]
		r_pop[i:,:] = pop
	return r_pop, r_t

def analytic_mean(t, params, pop):

	beta, gamma = params
	S, I = pop
	N = S + I

	diff = - gamma + beta
	mean = I * np.exp(diff * t)
	return mean

def main():

	params = np.array([ 0.05, 1])
	pop_0  = np.array([90,10])
	S = np.array([[-1,1],[1,-1]])

	M = 10
	T = 10
	sample_point = 100
	np.random.seed(42)
	pooling = np.empty((M, sample_point + 1))

	fig, ax = plt.subplots()
	fig.suptitle('SIS model')
	ax.set_title('I(0)=%s, S(0)=%s'%(pop_0[1], pop_0[0]))
	for nth in range(M):
		r_pop, r_t = gillespie_algorithm(params, hazard, S, pop_0, T, sample_point)
		pooling[nth, :] = r_pop[:,1]
		ax.plot(r_t, r_pop[:,1], '-', lw=1, alpha=0.5)

	mean = np.mean(pooling, axis=0)
	amean = analytic_mean(r_t, params, pop_0)
	ax.plot(r_t, mean, '.', lw=1, alpha=1, label='sample mean')
	ax.plot(r_t, amean, '-', lw=2, alpha=0.8, label='analytical mean')
	ax.set_xlabel('time points')
	ax.set_ylabel('number of infected individuals')
	ax.grid()
	ax.legend()
	plt.show()

if __name__ == "__main__":
	main()

