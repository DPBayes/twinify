# FAQ

### Can you tell me some details about the technical implementation?

twinify relies on [NumPyro](https://github.com/pyro-ppl/numpyro), a versatile probabilistic programming framework similar to [Pyro](http://pyro.ai/), for modelling and inference purposes. NumPyro uses fast CPU and GPU kernels for execution, which are provided by the [JAX](https://github.com/google/jax/) framework. Differentially private training routines for NumPyro are introduced by the [d3p](https://github.com/DPBayes/dppp) package.

### I'm unhappy with the quality of the generated data, what hyperparameters can I tweak?

First off, we need to warn you about **tweaking the hyperparameters** based on quality of the synthetic data: If you do that your choice will end up tailored to your specific data set which **can leak private information in subtle ways, degrading the privacy guarantees given by twinify**. Unfortunately, there's is no simple way to work around that other than finding good parameters on a similar public data set before working on your sensitive data.

If it is possible, you can usually improve quality of the synthetic data by relaxing your privacy constraints (i.e., choosing a larger ε for the same δ).

Also, differentially private learning is known to work better with more data. In case you are working with particularly small data set, you might need to collect more data in order to improve the utility of synthetic data.


### My data has lots of missing values, what do I need to do?
Real data is often incomplete and missing values might occur for a multitude of reasons, for example due to scarcity in measuring resources. twinify supports modelling features with missing values using a simple mechanism:
It assumes that values can be missing at random (independently from whether other feature values are missing as well) with a certain probability that is inferred from the data. During data generation, twinify first evaluates whether there should be a value, and, if so, samples one from the feature distribution specified in the model you provided.

Using automatic modelling, twinify detects and handles features with missing values automatically and you don't need to do anything. You can disable that behavior by setting the `--drop_na=1` command line argument to remove all data instances with missing values.

When writing your own NumPyro models, you can use the `twinify.na_model.NAModel` class to wrap around the feature distribution for achieving the same effect.

In mathematical terms, the likelihood of data in the `NAModel` is

![NAModelLikelihood](https://render.githubusercontent.com/render/math?math=p%28x%20%5Cmid%20q_%7BNA%7D%2C%20%5Ctheta_x%29%20%3D%20%5Cdelta_%7BNA%7D%28x%29%20q_%7BNA%7D%20%2B%20%5Cphi%28x%20%5Cmid%20%5Ctheta%29%281-q_%7BNA%7D%29%5Cmathbb%7B1%7D%28x%5Cneq%20NA%29)

where ![](https://render.githubusercontent.com/render/math?math=%5Cphi%28x%20%5Cmid%20%5Ctheta_x%29) is the likelihood of existing data x (according to the assigned feature distribution) and ![](https://render.githubusercontent.com/render/math?math=q_%7BNA%7D) denotes the probability that x is missing. Similar to other model parameters, twinify assigns a prior to and learns a posterior for ![](https://render.githubusercontent.com/render/math?math=q_%7BNA%7D).

### What distributions are supported in the automatic modelling?
Currently supported feature distributions are shown in the table below with the corresponding prior choices twinify uses for the parameters of these distributions.

| Distribution | Parameters           | Priors                          | Use for                         |
|--------------|----------------------|---------------------------------|---------------------------------|
| Normal       | location μ, scale σ  | μ ∼ 𝓝(0, 10),σ ∼ LogNormal(0,2) | (symmetric) continuous real numbers |
| Bernoulli    | probability p        | p ∼ Beta(1, 1)                  | binary categories (0/1 integers or "yes"/"no" strings) |
| Categorical  | probabilities **p**  | **p** ∼ Dirichlet(1, ..., 1)    | arbitrary categories (integer or string data) |
| Poisson      | rate λ               | λ ∼ Exp(1)                      | ordinal integer data |

### How does the automatic modelling work? What kind of model does it build?

As already mentioned, twinify's automatic modelling uses the distributions you specify for each feature (i.e., column in the data) to build a so called *mixture model* consisting of several *components*.  In each mixture component, the features are assumed to be independently modelled by the distributions you specified with component-specific parameters. Each data instance is associated with a single component with a probability given by the mixture's *weight*. During data generation, for each generated data instance, twinify first randomly picks a component according to the weights and then samples the data point according from the parameterised feature distributions in that component.

While all features are treated as independent in each mixture component, the mixture model as a whole is typically able to capture correlations between features.

In mathematical terms, the likelihood of the data given the model parameters for the mixture model is
![MixtureModelLikelihood](https://render.githubusercontent.com/render/math?math=p%28%5Cmathbf%7BX%7D%20%7C%20%5Cboldsymbol%7B%5Ctheta%7D%29%20%3D%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%5Cprod_%7Bd%3D1%7D%5ED%20%5Cphi_d%28%5Cmathbf%7BX%7D_%7B%3A%2Cd%7D%20%7C%5Cboldsymbol%7B%5Ctheta%7D_%7Bd%2Ck%7D%29)

where ![](https://render.githubusercontent.com/render/math?math=%5Cphi_d%28%5Cmathbf%7BX%7D_%7B%3A%2Cd%7D%20%7C%5Cboldsymbol%7B%5Ctheta%7D_%7Bd%2Ck%7D%29) is the density function of the user-defined feature distribution and ![](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BX%7D_%7B%3A%2Cd%7D) is the d-th feature column of the data set. To complete the probabilistic model twinify assigns non-informative prior distributions to the model parameters ![](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%5Ctheta_%7Bd%2Ck%7D) as well as the weights ![](https://render.githubusercontent.com/render/math?math=%5Cpi_k) for each of the K mixture components.

### What constraints does twinify set on NumPyro models?
There are only a few constraints twinify imposes. These are listed below.

You *must* define a function `model(x = None, num_obs_total = None)` containing the NumPyro model with the following constraints:

- `model` handles a single data instance at once and gets all data features in a single vector, i.e., `x` has shape `(num_features,)`.
- Feature values in `x` are ordered as they appear in the data set.
- `num_obs_total` is the number of total observations (i.e., the size of your data set) that you can use to scale the likelihood accordingly.
- During data generation, `x` and `num_obs_total` will both be `None`.
- `model` must return a sample for `x` with features following the same order as in the input.

You *may* specify a SVI `guide` function with the same arguments as `models`. If you do not, twinify uses NumPyro's automatic guides.

You *may* specify a preprocessing function `preprocess(loaded_data)` that gets the data as a `pandas.DataFrame` as it was loaded from the csv-file and returns a data frame which rows will be passed to `model` during inference. Your preprocessing may involve

- selecting the relevant feature columns out of the data
- reordering feature columns
- mapping strings to numeric values
- etc.

If you do not specify a preprocessing function, no preprocessing will take place and the loaded data is used as is.

You *may*  specify a post-processing function `postprocess(sampled_data)` that gets the data as a `pandas.DataFrame` sampled from the model after inference and returns a data frame to be written to the output csv-file. A possible post-processing step would be to map numeric values back to their string representation (i.e., reversing the mapping applied during preprocessing). If you do not specify a post-processing function, no post-processing will take place the generated data is stored as is.

### Can you tell me more about how the parameters affect privacy?

The private learning algorithm twinify uses is based on gradient descent optimization using perturbed gradients in every iteration. The gradients are first clipped so that their norm does not exceed a given threshold and then perturbed using Gaussian noise to mask any individuals contribution. Larger variance of Gaussian noise leads to more strict privacy guarantees, i.e., to smaller ε and δ.

twinify accepts the privacy level ε (and δ, typically determined automatically) as parameters and finds the variance for Gaussian noise to suffice this level of privacy. The noise variance is additionally affected by the number of epochs (Nₑ) and the subsampling ratio (q) as σ² ~= O(q Nₑ) since both affect the number of total iterations the algorithm performs and thus the number of times private data is handled.

Larger noise variance can negatively affect the learning so choosing too large values for q or Nₑ will likely give bad results.
