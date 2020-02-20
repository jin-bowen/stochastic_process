import numpy as np
import matplotlib.pyplot as plt

def hazard(r, n):

	return n * r

def gillespie_draw(r, hazard, Nt):

	# Compute propensities
	prob = hazard(r, Nt)
	
	# Compute time
	tau = np.random.exponential(1.0 / prob)
	
	return tau

def gillespie_algorithm(r, hazard, T, N_0, sample_point=1000):

	dt = float(T - 0.0)/float(sample_point)
	Nt = N_0

	nth_Nt = np.full(sample_point + 1, Nt)
	r_t   = np.linspace(0, T, num = sample_point + 1)

	cum_t = 0
	while(cum_t < T and Nt > 0):

		tau = gillespie_draw(r, hazard, Nt)
		cum_t += tau	

		i = int(np.floor(cum_t/dt))

		Nt += -1
		nth_Nt[i:] = Nt


	return nth_Nt, r_t

def analytical_sol(r, T, dt):

	beta_m, beta_p, gamma_p = r

	num_t = int(T/dt)

	m = np.empty(num_t)
	t = np.empty(num_t)
	for i in range(1, num_t):
		m_pre = m[i-1]
		m[i] = m_pre + (beta_m - m_pre) * dt
		t[i] = i * dt
	return m, t 

def analytic_mean_var(t, r, N_0):
	rt = -1 * r * t
	rt2 = -2 * r * t

	n2 = N_0 * N_0 

	mean = N_0 * np.exp(rt)
	var = N_0 * np.exp(rt) - N_0 * np.exp(rt2)

	return mean, var

def main():
	r = 0.3
	N_0  = 100
	M = 1000

	T = 5 * np.log(N_0) / r

	sample_point = 1000
	np.random.seed(42)
	pooling = np.empty((M, sample_point + 1))

	fig, ax = plt.subplots()
	for nth in range(M):
		nth_Nt, t = gillespie_algorithm(r, hazard, T, N_0, sample_point)
		pooling[nth, :] = nth_Nt
		ax.plot(t, nth_Nt, '-', lw=0.5, alpha=0.4)

	mean = np.mean(pooling, axis=0)
	var  = np.var(pooling, axis=0)
	ax.plot(t, mean, '.', lw=2, alpha=0.8, label='sample mean')
	ax.plot(t, var, '.', lw=2, alpha=0.8, label='sample variance')

	amean, avar = analytic_mean_var(t, r, N_0)
	ax.plot(t, amean, '-', lw=2, alpha=0.8, label='analytical mean')
	ax.plot(t, avar, '-', lw=2, alpha=0.8, label='analytical variance')

	ax.set_xlabel('time/arbitrary unit')
	ax.set_ylabel('number of particle A')
	fig.suptitle('A -> None')
	ax.set_title('r=%s, # of simluation=%s'%(r,M))
	ax.legend()
	ax.grid()
	plt.show()

if __name__ == "__main__":
	main()

