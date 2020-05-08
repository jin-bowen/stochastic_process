import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def hazard(params, pop):

	alpha, beta = params
	NA, NB = pop

	return np.array([ alpha * NA,
			beta * NB])

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

def analytic_mean_var(t, params, pop_0):

	NA, NB = pop_0
	alpha, beta = params

	N = NA + NB
	k1 = alpha + beta
	k2 = alpha * N/ (k1) 
	betan = alpha * N
	n2beta = 2.0 * N * alpha - alpha + beta
	inv = 1.0/k1

	A = betan + n2beta * betan / k1 
	A *= 0.5 * inv

	B = n2beta * (NB - betan / k1)
	B *= inv

	C = NB * NB - A - B

	mean = k2 - (k2 - NB) * np.exp(-1 * k1 * t)
	meansq = mean * mean
	var = A + B * np.exp(-1 * k1 * t) + C * np.exp(-2 * k1 * t)
	var -= meansq

	return mean, var

def main():

	params = np.array([ 0.2, 0.1 ])
	pop_0  = np.array([90,10])
	S = np.array([[-1,1],[1,-1]])

	M = 100
	T = np.log(90) / 0.2
	sample_point = 100
	np.random.seed(42)
	pooling = np.empty((M, sample_point + 1))

	fig, ax = plt.subplots()
	fig.suptitle('A = B')
	for nth in range(M):
		r_pop, r_t = gillespie_algorithm(params, hazard, S, pop_0, T, sample_point)
		pooling[nth, :] = r_pop[:,1]
		ax.plot(r_t, r_pop[:,1], '-', lw=1, alpha=0.2)

	mean = np.mean(pooling, axis=0)
	var  = np.var(pooling, axis=0)

	ax.plot(r_t, mean, '.', lw=1, alpha=0.6, label='sample mean')
	ax.plot(r_t, var, '.', lw=1, alpha=0.6, label='sample variance')

	amean, avar = analytic_mean_var(r_t, params, pop_0)
	ax.plot(r_t, amean, '-', lw=2, alpha=0.8, label='analytical mean')
	ax.plot(r_t, avar, '-', lw=2, alpha=0.8, label='analytical variance')

	ax.set_xlabel('time points')
	ax.set_ylabel('number of particle B')
	ax.legend()
	plt.show()

if __name__ == "__main__":
	main()

