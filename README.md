JAX implementation of a preconditioned Sequential Monte Carlo framework for Bayesian inference. It transforms samples from an initial distribution to a target posterior through reweighting, resampling, and mutation steps. To improve sampling efficiency a t-preconditioned Crank-Nicolson MCMC algorithm was used. The framework supports just-in-time compilation, automatic differentiation, and vectorization, making it suitable for computationally expensive inference problems. Numerical experiments, including applications to gravitational-wave inference, show that the sampler efficiently captures multimodal target distributions.

Inspired by related work on Preconditioned Monte Carlo, persistent sampling, and preconditioned Crank–Nicolson kernels:

- [Validating Sequential Monte Carlo for Gravitational-Wave Inference](https://arxiv.org/abs/2506.18977)

- [Markov Chain Monte Carlo and Variational Inference: Bridging the Gap](https://arxiv.org/abs/1410.6460)

- [Accelerating astronomical and cosmological inference with Preconditioned Monte Carlo](https://arxiv.org/abs/2207.05652)

- [pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology](https://arxiv.org/abs/2207.05660)

- [blackjax](https://github.com/blackjax-devs/blackjax)

The sampler is now using dummy normalizing flow which will be substituted with a real one in the near future. Also changes in the code are expected. 
