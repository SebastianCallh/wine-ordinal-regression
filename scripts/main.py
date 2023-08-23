#!/usr/bin/env python3

import os
import pandas as pd
import numpyro as npo
from numpyro.infer import MCMC, NUTS, Predictive

from numpyro.infer.reparam import TransformReparam
from numpyro import distributions as dist
import arviz as az

import jax
from jax.random import PRNGKey
from jax import numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

num_chains = 4
npo.set_host_device_count(num_chains)


def save_fig(fig, name, fig_root="plots"):
    os.makedirs(fig_root, exist_ok=True)
    fig.savefig(os.path.join(fig_root, name))


df_raw = pd.read_csv(os.path.join("data", "wine-quality-white-and-red.csv"))

## Clean out some outliers
df = df_raw[
    (df_raw["density"] < 1.01)
    & (df_raw["free sulfur dioxide"] < 200)
    & (df_raw["citric acid"] < 1.1)
]
pairgrid = sns.pairplot(df, hue="type", corner=True)
save_fig(pairgrid, "data_pairplot.png")


fig, ax = plt.subplots()
sns.countplot(df, x="type", ax=ax)
ax.set_title("Wine type count")
save_fig(fig, "wine_type_count.png")

fig, ax = plt.subplots()
sns.countplot(df, x="quality", ax=ax)
ax.set_title("Wine quality distribution")
save_fig(fig, "quality_distribution.png")


## Prepare features and inference
def inference(model, num_warmup, num_samples, num_chains, **kwargs):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    mcmc.run(rng_key, **kwargs)
    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    post_predictive = Predictive(model, samples)
    post_pred_samples = post_predictive(
        rng_key, **{k: v for k, v in kwargs.items() if k != "y"}
    )
    idata = az.from_numpyro(mcmc, posterior_predictive=post_pred_samples)

    y_mean_pred = idata.posterior_predictive["y"].mean(("chain", "draw")).values
    idata.posterior_predictive["residual"] = y - y_mean_pred
    idata.posterior["cutpoints"] = (
        ("chain", "draw", "sample"),
        jax.nn.sigmoid(idata.posterior.c.values),
    )

    return idata


def ordinal_regression(X, t, ntypes, concentration, anchor_point=0.0, y=None):
    N, D = X.shape

    alpha_loc = npo.sample("alpha_loc", dist.Cauchy(0, 1))
    alpha_scale = npo.sample("alpha_scale", dist.HalfCauchy(1))

    beta_loc = npo.sample("beta_loc", dist.Cauchy(jnp.zeros(D), 1)).reshape(-1, 1)
    beta_scale = npo.sample("beta_scale", dist.HalfCauchy(jnp.ones(D))).reshape(-1, 1)

    with npo.plate("type", ntypes, dim=-1):
        alpha = npo.sample("alpha", dist.Normal(alpha_loc, alpha_scale))

        with npo.plate("dim", D, dim=-2):
            beta = npo.sample("beta", dist.Normal(beta_loc, beta_scale))

    with npo.handlers.reparam(config={"c": TransformReparam()}):
        c = npo.sample(
            "c",
            dist.TransformedDistribution(
                dist.Dirichlet(concentration),
                dist.transforms.SimplexToOrderedTransform(anchor_point),
            ),
        )

    with npo.plate("obs", N):
        eta = alpha[t] + jnp.sum(X.T * beta[:, t], 0)
        npo.sample("y", dist.OrderedLogistic(eta, c), obs=y)


X = jnp.asarray(df.drop(columns=["quality", "type"]))
X_mean = X.mean(0)
X_std = X.std(0)
X = (X - X_mean) / X_std
t = jnp.asarray(df["type"].apply(lambda x: 0 if x == "white" else 1))
ntypes = df["type"].nunique()

y = jnp.asarray(df["quality"])
y_min = y.min()
y = y - y_min

nclasses = df["quality"].nunique()
concentration = jnp.ones((nclasses,))
rng_key = PRNGKey(1234)

idata = inference(
    ordinal_regression,
    X=X,
    y=y,
    t=t,
    ntypes=ntypes,
    concentration=concentration,
    num_warmup=500,
    num_samples=1000,
    num_chains=num_chains,
)

axs = az.plot_forest(idata, var_names="beta")
save_fig(axs[0].get_figure(), "posterior_betas.png")

ax = az.plot_ppc(idata)
save_fig(ax.get_figure(), "posterior_predictive_check.png")

fig, ax = plt.subplots()
kwargs = dict(alpha=0.6, bins=30)
ax.hist(
    idata.posterior_predictive["residual"].values,
    label="Vanilla",
    alpha=0.6,
    bins=30,
)
ax.set_title("Residuals")
ax.set_xlabel("y - E[p(y|theta, data)]")
ax.legend()

idata.posterior_predictive["MAE"] = jnp.mean(
    jnp.abs(idata.posterior_predictive["residual"].values)
).item()

idata.posterior["cutpoints"] = (
    ("chain", "draw", "sample"),
    jax.nn.sigmoid(idata.posterior.c.values),
)

c_mean = jnp.append(idata.posterior.cutpoints.mean(dim=("draw", "chain")).values, 1)
c_hdi = az.hdi(idata.posterior.cutpoints.values)
c_hdi = jnp.concatenate((c_hdi, jnp.ones((1, 2))))
c_lower, c_upper = c_hdi[:, 0], c_hdi[:, 1]

scores = jnp.arange(nclasses) + y_min

fig, ax = plt.subplots()
ax.bar(scores, c_mean, yerr=(c_mean - c_lower, c_upper - c_mean))
ax.set_title("Posterior cutpoints")
ax.set_xlabel("Quality")
ax.set_ylabel("Cumulative density")
save_fig(ax.get_figure(), "posterior_cutpoints.png")

idata.to_netcdf(os.path.join("data", "idata.nc"))
