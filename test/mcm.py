import numpy as np
import emcee
import matplotlib.pyplot as plt

def log_prob(x):
    return -0.5 * x[0]**2

ndim = 1
nwalkers = 10
nsteps = 5000
burnin = 1000

initial_pos = np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(initial_pos, nsteps, progress=True)

samples = sampler.get_chain(discard=burnin, flat=True)

assert samples.shape[1] == ndim, "Unexpected sample dimension"
assert samples.shape[0] > 2000, "Too few samples"

mean_est = np.mean(samples)
var_est = np.var(samples)

assert abs(mean_est) < 0.1, f"Mean too far from 0: {mean_est}"
assert 0.8 < var_est < 1.2, f"Variance out of range: {var_est}"
assert np.all(np.abs(samples) < 10), "Samples out of expected range"

print("All sanity checks passed.")

plt.figure(figsize=(8, 5))
plt.hist(samples, bins=50, density=True, alpha=0.7, label="MCMC samples")

x = np.linspace(-4, 4, 200)
plt.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-0.5 * x**2), "r-", lw=2, label="N(0,1)")

plt.xlabel("x")
plt.ylabel("Density")
plt.title("MCMC test with emcee (with asserts)")
plt.legend()
plt.show()
