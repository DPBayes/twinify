# Twinify
Twinify is a software package for **privacy-preserving generation of a synthetic twin** to a given sensitive **data set**.

Twinify relies on [NumPyro](https://github.com/pyro-ppl/numpyro), a versatile probabilistic programming framework similar to [Pyro](http://pyro.ai/), for modelling and inference purposes. NumPyro uses fast CPU and GPU kernels for execution, which are provided by the [JAX](https://github.com/google/jax/) framework. Differentially private training routines for NumPyro are introduced by the [d3p](https://github.com/DPBayes/dppp) package.

Twinify implements the differentially private data sharing process introduced by [Jälkö et al.](https://arxiv.org/pdf/1912.04439.pdf) and offers automatic modelling for easy building of models fitting the data. If you are already experienced with NumPyro you can also implement your own model directly.

![PeopleBeforeImage](https://dpbayes.github.io/twinify/PeopleBefore.jpg)
![PeopleAfterImage](https://dpbayes.github.io/twinify/PeopleAfter.jpg)

## The Differentially Private Data Sharing Workflow

Often data that could be very useful for the scientific community is subject to privacy regulations and cannot be shared. Differentially private data sharing allows to generate synthetic data that is statistically similar to the original data - the `synthetic twin` - while at the same time satisfying a mathematical privacy formulation known as [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy). Differential privacy measures the level of privacy in terms of positive parameters ε and δ - where smaller values imply stronger privacy - thus giving us concrete knobs to tune the synthetic data generation to our privacy needs and ensuring that private information remains private!

In order to generate data, we rely on [probabilistic modelling](https://en.wikipedia.org/wiki/Category:Probabilistic_models), which means we assume the data follows a probability distribution with some parameters, which we can infer privately. In order to generate the synthetic twin data, we sample from this distribution with the learned parameters, the *posterior predictive distribution*.

As an example, consider a population of individuals with varying height shown in the first panel of the illustrations above. We present the heights as a histogram in panel (a) of the figure below. We then fit a probabilistic model for this data, the blue curve in in panel (b), and sample new data from this distribution magenta dots in (c).

![ProbabilisticModellingStrip](https://dpbayes.github.io/twinify/ProbabilisticModellingStrip.jpg)


As the learning of the model is done under differential privacy, the sampled data preserves the anonymity of individuals while maintaining the statistical properties of the original population. This is shown in the second panel in the illustration above.

## Installing Twinify

Install Twinify from the Python Package Index using `pip`

```pip install twinify```

or clone the repository

```
git clone https://github.com/DPBayes/twinify
cd twinify
pip install .
```

## Using Twinify
You can use Twinify for arbitrary tabular data sets. The main thing you need to do is to set up the probabilistic model for which we fit the data. Twinify supports automatic model building for users with less experience in programming and probabilistic modelling but you can also implement a full-blown model using NumPyro.

### Automatic Modelling
Twinifys automatic modelling feature builds a mixture model for user specified feature distributions. You can set up a text file, in which you only need to specify a single distribution for each of your features. A feature is identified by the full column name in the data set csv-file. For a data set containing features `Age` and `Height (cm)` the model file might look like

```
Age        : Poisson
# you can also have comments in here
Height (cm): Normal
```

A example of such text file for a larger data set is available in `examples/covid19_analysis/models/full_model.txt`. Automatic modelling also automates the encoding of string valued features into suitable domain.

### Building Models in NumPyro
If you are familiar with NumPyro and want a more flexible way of specifying models, you can provide a Python file containing NumPyro code to Twinify. All you need to do is providing a `model` function that specifies the NumPyro model for a single data instance `x`. You also have to define functions for pre- and postprocessing of data (if required). You can find details on the exact requirements for NumPyro models in the FAQ and an example in `examples/covid19_analysis/models/numpyro_model_example.py`.

### How to Run Twinify
Once you have have set the probabilistic model, you can run Twinify by calling

```
python twinify.py input_data_path model_path output_path_prefix
```

where the model can be specified either as the text file for automatic modelling or as a python module that contains the NumPyro model.

Twinify will output the generated synthetic data as `output_path_prefix.csv`, a file with learned model parameters as `output_path_prefix.p`
and, optionally, plots visualizing summary characteristics of the generated data as `output_path_prefix_missing_value_plots.svg`, `output_path_prefix_marginal_plots.svg` and `output_path_prefix_correlation_plots.svg`.

There is a number of optional command line arguments that further influence Twinify's behaviour:

- `--epsilon` - Privacy parameter ε (positive real number): Use this argument to specify the $\epsilon$ privacy level. Smaller is better (but may negatively impact utility). In general values less than one are considered strong privacy and values less than 2 still reasonable.
- `--delta` - Privacy parameter δ (positive real number): Use this argument to override the default choice for $\delta$ (should rarely be required). Smaller is better. Values larger than 1/N, where N is the size of your data set, are typically considered unsafe.
- `--num_synthetic` - Number of synthetic samples (integer): Use this to set how many samples you want from the generative model. This has no effect on the privacy guarantees for the synthetic data.
- `--k` - Number of mixture components (integer): Use this argument to set the number of mixture components when automatic modelling is used. A reasonable choice would be of same magnitude as the number of features.
- `--sampling_ratio`, `-q` - Subsampling ratio (real number): Use this argument to set the relative size of subsets (batches) of data the iteratively private learning is uses. This has privacy implications and is further discussed in FAQ.
- `--num_epochs`,`-e`, - Number of learning epochs (integer): Use this argument to set the number of passes through the data (*epochs*) the private learning performs. This has privacy implications and is further discussed in FAQ.
- `--clipping_threshold` - Privacy parameter (positive real number): Use this argument to adapt the clipping of gradients, an internal parameter for the private learning that limits how much each sample can effect the learning. It is only advised for experienced users to change this parameter.
- `--seed` - Stochasticity seed (integer): Use this argument to seed the initial random state to fix internal stochasticity of Twinify if you need reproducibility. If not given, Twinify will use a strong source of randomness.
- `--drop_na` - Preprocessing behavior (0 or 1): Use this argument to remove (1) or keep (0) data instances with missing values.
- `--visualize` - Visualization behavior ("none", "store", "popup" or "both"): Use this argument to control whether Twinify provides a visualization of summary characteristics of the generated data. Options are:
    - `none`: Disable visualization.
    - `store`: Store the plots as files.
    - `popup`: Show the plots in popup windows after sampling.
    - `both`: Store the plots as files and show popup windows.

As an example, say we have data in `my_data.csv` and a model description for automatic modelling in `my_model.txt`. We want 1000 samples of generated data to be stored in `my_twin.csv` and fix Twinify's internal randomness with a seed for reproducibility. We also want to store the plots of summary characteristic visualizations but not display them at runtime. This is how we run Twinify:

```
python twinify.py my_data.csv my_model.txt my_data --seed=123 --num_synthetic=1000 --visualize=store
```

In the case that we wrote a model with NumPyro instead of relying on Twinify's automatic modelling, our call would like like

```
python twinify.py my_data.csv my_numpyro_model.py my_data --seed=123 --num_synthetic=1000 --visualize=store
```

## Technical detail FAQ:

### I'm unhappy with the quality of the generated data, what hyperparameters can I tweak?

First off, we need to warn you about **tweaking the hyperparameters** based on quality of the synthetic data: If you do that your choice will end up tailored to your specific data set which **can leak private information in subtle ways, degrading the privacy guarantees given by Twinify**. Unfortunately, there's is no simple way to work around that other than finding good parameters on a similar public data set before working on your sensitive data.

If it is possible, you can usually improve quality of the synthetic data by relaxing your privacy constraints (i.e., choosing a larger ε for the same δ).

Also, differentially private learning is known to work better with more data. In case you are working with particularly small data set, you might need to collect more data in order to improve the utility of synthetic data.


### My data has lots of missing values, what do I need to do?
Real data is often incomplete and missing values might occur for a multitude of reasons, for example due to scarcity in measuring resources. Twinify supports modelling features with missing values using a simple mechanism:
It assumes that values can be missing at random (independently from whether other feature values are missing as well) with a certain probability that is inferred from the data. During data generation, Twinify first evaluates whether there should be a value, and, if so, samples one from the feature distribution specified in the model you provided.

Using automatic modelling, Twinify detects and handles features with missing values automatically and you don't need to do anything. You can disable that behavior by setting the `--drop_na=1` command line argument to remove all data instances with missing values.

When writing your own NumPyro models, you can use the `twinify.na_model.NAModel` class to wrap around the feature distribution for achieving the same effect.

In mathematical terms, the likelihood of data in the `NAModel` is

![NAModelLikelihood](https://render.githubusercontent.com/render/math?math=p%28x%20%5Cmid%20q_%7BNA%7D%2C%20%5Ctheta_x%29%20%3D%20%5Cdelta_%7BNA%7D%28x%29%20q_%7BNA%7D%20%2B%20%5Cphi%28x%20%5Cmid%20%5Ctheta%29%281-q_%7BNA%7D%29%5Cmathbb%7B1%7D%28x%5Cneq%20NA%29)

where ![](https://render.githubusercontent.com/render/math?math=%5Cphi%28x%20%5Cmid%20%5Ctheta_x%29) is the likelihood of existing data x (according to the assigned feature distribution) and ![](https://render.githubusercontent.com/render/math?math=q_%7BNA%7D) denotes the probability that x is missing. Similar to other model parameters, Twinify assigns a prior to and learns a posterior for ![](https://render.githubusercontent.com/render/math?math=q_%7BNA%7D).

### What distributions are supported in the automatic modelling?
Currently supported feature distributions are shown in the table below with the corresponding prior choices Twinify uses for the parameters of these distributions.

| Distribution | Parameters           | Priors                        | Use for                         |
|--------------|----------------------|-------------------------------|---------------------------------|
| Normal       | location μ, scale σ  | μ ∼ 𝓝(0,1),σ ∼ LogNormal(0,2) | (symmetric) continuous real numbers |
| Bernoulli    | logit-probability z  | z ∼ 𝓝(0, 1)                   | binary categories (0/1 integers or "yes"/"no" strings) |
| Categorical  | probabilities **p**      | **p** ∼ Dirichlet(1, ..., 1)      | arbitrary categories (integer or string data) |
| Poisson      | rate λ               | λ ∼ Exp(1)                    | ordinal integer data |

### How does the automatic modelling work? What kind of model does it build?

As already mentioned, Twinify's automatic modelling uses the distributions you specify for each feature (i.e., column in the data) to build a so called *mixture model* consisting of several *components*.  In each mixture component, the features are assumed to be independently modelled by the distributions you specified with component-specific parameters. Each data instance is associated with a single component with a probability given by the mixture's *weight*. During data generation, for each generated data instance, Twinify first randomly picks a component according to the weights and then samples the data point according from the parameterised feature distributions in that component.

While all features are treated as independent in each mixture component, the mixture model as a whole is typically able to captures correlations between features.

In mathematical terms, the likelihood of the data given the model parameters for the mixture model is
![MixtureModelLikelihood](https://render.githubusercontent.com/render/math?math=p%28%5Cmathbf%7BX%7D%20%7C%20%5Cboldsymbol%7B%5Ctheta%7D%29%20%3D%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%5Cprod_%7Bd%3D1%7D%5ED%20%5Cphi_d%28%5Cmathbf%7BX%7D_%7B%3A%2Cd%7D%20%7C%5Cboldsymbol%7B%5Ctheta%7D_%7Bd%2Ck%7D%29)

where ![](https://render.githubusercontent.com/render/math?math=%5Cphi_d%28%5Cmathbf%7BX%7D_%7B%3A%2Cd%7D%20%7C%5Cboldsymbol%7B%5Ctheta%7D_%7Bd%2Ck%7D%29) is the density function of the user-defined feature distribution and ![](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BX%7D_%7B%3A%2Cd%7D) is the d-th feature column of the data set. To complete the probabilistic model Twinify assigns non-informative prior distributions to the model parameters ![](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%5Ctheta_%7Bd%2Ck%7D) as well as the weights ![](https://render.githubusercontent.com/render/math?math=%5Cpi_k) for each of the K mixture components.

### What constraints does Twinify set on NumPyro models?
There are only a few constraints Twinify imposes. These are listed below.

You *must* define a function `model(x, num_obs_total)` containing the NumPyro model with the following constraints:

- `model` handles a single data instance at once and gets all data features in a single vector, i.e., `x` has shape `(num_features,)`.
- Feature values in `x` are ordered as they appear in the data set.
- `num_obs_total` is the number of total observations (i.e., the size of your data set), that you can use to scale the likelihood accordingly
- `model` needs contain a sample site called `x` that samples a full data instance, similar to argument `x`. You can use `twinify.interface.sample_combined` as a quick way to combine samples of features you modelled by separate distributions.

You *may* specify a SVI `guide` function with the same arguments as `models`. If you do not, Twinify uses NumPyro's automatic guides.

You *may* specify a preprocessing function `preprocess(loaded_data)` that gets the data as a `pandas.DataFrame` as it was loaded from the csv-file and returns a data frame which rows will be passed to `model` during inference. Your preprocessing may involve

- selecting the relevant feature columns out of the data
- reordering feature columns
- mapping strings to numeric values
- etc.

If you do not specify a preprocessing function, no preprocessing will take place and the loaded data is used as is.

You *may*  specify a post-processing function `postprocess(sampled_data)` that gets the data as a `pandas.DataFrame` sampled from the model after inference and returns a data frame to be written to the output csv-file. A possible post-processing step would be to map numeric values back to their string representation (i.e., reversing the mapping applied during preprocessing). If you do not specify a post-processing function, no post-processing will take place the generated data is stored as is.

### Can you tell me more about how the parameters affect privacy?

The private learning algorithm Twinify uses is based on gradient descent optimization using perturbed gradients in every iteration. The gradients are first clipped so that their norm does not exceed a given threshold and then perturbed using Gaussian noise to mask any individuals contribution. Larger variance of Gaussian noise leads to more strict privacy guarantees, i.e., to smaller ε and δ.

Twinify accepts the privacy level ε (and δ, typically determined automatically) as parameters and finds the variance for Gaussian noise to suffice this level of privacy. The noise variance is additionally affected by the number of epochs (Nₑ) and the subsampling ratio (q) as σ² ~= O(q Nₑ) since both affect the number of total iterations the algorithm performs and thus the number of times private data is handled.

Larger noise variance can negatively affect the learning so choosing too large values for q or Nₑ will likely give bad results.
