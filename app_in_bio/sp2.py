import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def hazard(params, pop):

	ra,rb,ka,kb,ua,ub = params
	xa, xb = pop

	k1 = (ka)/(ka + 2*xa + xb)
	k2 = (kb)/(kb + 2*xb + xa)

	da = ra * k1 * xa
	db = rb * k2 * xb

	return np.array([da,ua*xa,db,ub*xb])

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

	params = np.array([ 3, 4, 50, 40, 1, 1])
	pop_0  = np.array([5,5])
	S = np.array([[1,-1,0,0],[0,0,1,-1]])

	M = 5
	T = 100
	sample_point = 1000
	np.random.seed(42)

	fig, ax = plt.subplots()
	fig.suptitle('system2')
	for nth in range(M):
		r_pop, r_t = gillespie_algorithm(params, hazard, S, pop_0, T, sample_point)
		ax.plot(r_t, r_pop[:,0], '-', lw=1)

	ax.set_xlabel('time points')
	ax.set_ylabel('number of particle A')
	ax.grid(True)
	ax.legend()
	plt.savefig('system2_SP.png')

if __name__ == "__main__":
	main()

