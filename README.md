## **Package description**
---

`jaxpsmc` implements a Sequential Monte Carlo sampler in JAX for Bayesian posterior approximation and marginal-likelihood estimation. The implementation follows the standard SMC construction: it represents the target distribution by a set of weighted particles, introduces a sequence of intermediate distributions between prior and posterior, and applies reweighting, resampling, and mutation to control weight degeneracy while transporting the particle population toward the posterior.

The sampler evolves particles along a temperature-annealed sequence of intermediate distributions, beginning at the prior and ending at the posterior. This allows the sampler to explore effectively difficult targets, especially when the posterior is multimodal. At small $\beta$, the likelihood is flattened and the particles explore a broader approximation to the posterior geometry; as $\beta$ increases, the distribution contracts toward the true posterior. The sampler is intended for nonlinear geometry, bounded coordinates, periodic or reflective parameters, and posteriors with well-separated modes.

At the algorithmic level, `jaxpsmc` consists of the following components.

1. **Reweighting:** at a proposed inverse temperature $\beta_t$, the algorithm updates particle weights using the likelihood increment, evaluates either the effective sample size or the unique sample size, and then selects the next temperature by bisection so that the chosen degeneracy criterion remains close to a target. This makes the annealing steps data-adaptive.

2. **Resampling:** after reweighting, the weighted particle set is resampled using either multinomial or systematic resampling. This prevents the approximation from collapsing onto a small number of particles with large weight. 

3. **Mutation:** the current implementation mutates the active particles using a t-preconditioned Crank-Nicolson kernel in latent space. This is the stage at which local exploration occurs. It is also the stage at which preconditioning takes place: if the target can be mapped to a space with less correlated structure, even a simple local proposal can become substantially more effective. Integrating a normalizing flow for preconditioning is still work in progress, and the current implementation uses an identity flow.



## **Examples**
---

At present moment there are two reproducible examples: numerical experiments and gravitational-wave validation. To see the experiments set-up and results please visit [jaxpsmc-sampling_examples](https://github.com/jaxpsmc/jaxpsmc-sampling_examples)



## Literature
---

1. **Compiling machine learning programs via high-level tracing**
   Frostig, R., Johnson, M. J., Leary, C. (2018).
   Link: https://research.google/pubs/compiling-machine-learning-programs-via-high-level-tracing/

2. **JAX: composable transformations of Python+NumPy programs**
   Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018).
   Link: https://github.com/jax-ml/jax

3. **Sequential Monte Carlo samplers** 
   Del Moral, P., Doucet, A., Jasra, A. (2006).  
   Link: https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf

4. **Sequential Monte Carlo for Bayesian Computation**  
   Del Moral, P., Doucet, A., & Jasra, A. (2008).  
   Link: https://people.bordeaux.inria.fr/pierre.delmoral/valencia.pdf

5. **Annealed importance sampling**  
   Neal, R. M. (1998).  
   Link: https://arxiv.org/pdf/physics/9803008

6. **MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster**  
   Cotter, S. L., Roberts, G. O., Stuart, A. M., White, D. (2013).  
   Link: https://eprints.maths.manchester.ac.uk/2215/1/1202.0709v3.pdf

7. **Accelerating astronomical and cosmological inference with Preconditioned Monte Carlo**  
   Karamanis, M., Beutler, F., Peacock, J. A., Nabergoj, D., Seljak, U. (2022).  
   Link: https://arxiv.org/abs/2207.05652

8. **Rethinking the Effective Sample Size**  
   Elvira, V., Martino, L., Robert, C. P. (2022).  
   Link: https://arxiv.org/abs/1809.04129

9. **The EM algorithm**  
   Peel, D., McLachlan, G. J. (2000).  
   Link: https://www.econstor.eu/handle/10419/22198

10. **Stan: A Probabilistic Programming Language**  
   Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Riddell, A. (2017).  
   Journal of Statistical Software, 76(1), 1–32.  
   Link: https://www.jstatsoft.org/article/view/v076i01

11. **Validating Sequential Monte Carlo for Gravitational-Wave Inference**  
   Williams, M. J., Karamanis, M., Luo, Y., & Seljak, U. (2025).   
   Link: https://arxiv.org/abs/2506.18977
