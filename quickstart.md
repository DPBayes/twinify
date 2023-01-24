# Getting started

## Using twinify

twinify can be used as a software library from your own application or as a stand-alone command line tool operating on data sets provided as a [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values). Either way, the high-level steps are the same and we outline them in the following for the command line tool. You can find a brief overview of twinify's API for library use further below.

### Choosing the Method
The first thing you need to do is decide whether you want to use the NAPSU-MQ approach or learn a probabilistic model using DPVI.

- **NAPSU-MQ** learns a maximum entropy distribution that best reproduces a user-chosen set of marginal queries on the data. NAPSU-MQ produces a model that encapsulates the additional uncertainty introduced by differential privacy. However, currently it is only suitable for fully categorical data. May exhibit long runtimes for data sets with many feature dimensions.
- **DPVI** is capable of learning any probabilistic model you specify, for categorical, continuous or mixed data. However, the result is only an approximation to the true posterior and it is unable to explicitly capture additional uncertainty due to differential privacy.

If you have fully categorical data, you will likely obtain better results with **NAPSU-MQ**. However, if your data has a large number of feature dimensions, you may find that you can get acceptable results in shorter time using **DPVI**.

If your data contains non-categorical features, **DPVI** is your only choice without resorting to discretization. **DPVI** might also be an interesting option if you have strong data-independent prior knowledge that you want to incorporate into your model.

### Defining the Model
The main thing you need to do next for either method is to define the probabilistic model to be learned. The following describes the modelling approaches for the different methods, assuming an input csv file with three features that are titled `Age` and `Height (cm)` and `Eye color`.

#### NAPSU-MQ: Defining Marginal Queries
For NAPSU-MQ this means that you must specify the the marginal queries to preserve. You can in principle select any number of queries with any subset of features, however, the larger the number of queries, the longer the fitting of the model will take.

To specify marginal queries, you have to create a text file in which you list one query per line and all features covered by the query using the corresonding column name in the data csv file, separated by commas.

We assume here that the features Age and Height are discretized and require NAPSU-MQ to fit all feature marginals as well as the two-way marginal over the combined features Age and Height, resulting in the following model/query file:

```
Age
Height (cm)
Age, Height (cm)
Eye color
```

#### DPVI: Automatic Modelling
twinifys automatic modelling feature for DPVI builds a mixture model for user specified *feature distributions*. Technically speaking, the feature distribution specifies the distribution of the feature conditioned on the latent mixture component assignment. Under this conditioning, feature distributions are assumed to be independent.

To specify the feature distributions, you have to create a text file in which you only need to specify a single distribution for each of your features. For the assumed example the model file might look like:

```
Age        : Poisson
# you can also have comments in here
Height (cm): Normal
Eye color  : Categorical
```

A example of such text file for a larger data set is available in `examples/covid19_analysis/models/full_model.txt`. In automatic modelling twinify chooses a suitable non-/weakly informative prior for the parameters of the feature distribution. It also automates the encoding of string valued features into a suitable domain according to the chosen feature distribution.

#### DPVI: Building Models in NumPyro
If you are familiar with the NumPyro probabilistic programming framework and want a more flexible way of specifying models, you can provide a Python file containing NumPyro code to twinify. All you need to do is define a `model` function that specifies the NumPyro model for a single data instance `x`. You also have to define functions for pre- and postprocessing of data (if required). You can find details on the exact requirements for NumPyro models in the FAQ below and an example in `examples/covid19_analysis/models/numpyro_model_example.py`.