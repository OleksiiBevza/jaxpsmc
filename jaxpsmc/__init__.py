"""
Public API for jaxpsmc
"""

from .prior_jax import Prior, NORMAL, UNIFORM
from .sampler.sampler_jax import (
    SamplerJAX,
    SamplerConfigJAX,
    RunOutputJAX,
    IdentityFlowJAX,
)
from .sampler.posterior_jax import posterior_jax, PosteriorOut
from .particles_jax import ParticlesState, ParticlesStep, compute_results_jax
from .geometry.dili_geometry_jax import DILIPCNGeometry, build_dili_pcn_geometry_jax

__version__ = "0.1.0"

__all__ = [
    "Prior",
    "NORMAL",
    "UNIFORM",
    "SamplerJAX",
    "SamplerConfigJAX",
    "RunOutputJAX",
    "IdentityFlowJAX",
    "posterior_jax",
    "PosteriorOut",
    "ParticlesState",
    "ParticlesStep",
    "compute_results_jax",
    "DILIPCNGeometry",
    "build_dili_pcn_geometry_jax",
]
