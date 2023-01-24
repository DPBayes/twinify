Command line interface
========================

Once you have have set the probabilistic model, you can run twinify by calling from your command line::

    twinify [napsu|vi] input_data_path model_path output_path_prefix


where the model is specified as

- NAPSU-MQ: text file containing marginal queries
- DPVI: either the text file for automatic modelling or as a python module that contains the NumPyro model.

twinify will output the generated synthetic data as ``output_path_prefix.csv`` and a file with learned model parameters as ``output_path_prefix.p``.

There are a number of (optional) command line arguments that further influence twinify's behaviour:

- ``--epsilon``:  Privacy parameter ε (positive real number): Use this argument to specify the ε privacy level. Smaller is better (but may negatively impact utility). In general values less than 1 are considered strong privacy and values less than 2 still reasonable.
- ``--delta``:  Privacy parameter δ (positive real number between 0 and 1): Use this argument to override the default choice for δ (should rarely be required). Smaller is better. Recommended to be less than 1/N, where N is the size of your data set. Values larger are typically considered unsafe.
- ``--num_synthetic``:  Number of synthetic samples (integer): Use this to set how many samples you want from the generative model. This has no effect on the privacy guarantees for the synthetic data.

- ``--seed``:  Stochasticity seed (integer): Use this argument to seed the initial random state to fix internal stochasticity of twinify *if you need reproducibility*. **twinify will use a strong source of randomness by default** if this argument is not given.
- ``--drop_na``:  Preprocessing behavior: Use this flag to remove any data instances with at least one missing value.

Command line arguments specific to DPVI (ignored by NAPSU-MQ):

- ``--k``:  Number of mixture components (integer): Use this argument to set the number of mixture components when automatic modelling is used. A reasonable choice would be of same magnitude as the number of features.
- ``--sampling_ratio``, ``-q``:  Subsampling ratio (real number between 0 and 1): Use this argument to set the relative size of subsets (batches) of data the iteratively private learning is uses. This has privacy implications and is further discussed in FAQ.
- ``--num_epochs``, ``-e``:  Number of learning epochs (integer): Use this argument to set the number of passes through the data (*epochs*) the private learning performs. This has privacy implications and is further discussed in FAQ.
- ``--clipping_threshold``:  Privacy parameter (positive real number): Use this argument to adapt the clipping of gradients, an internal parameter for the private learning that limits how much each sample can effect the learning. It is only advised for experienced users to change this parameter.

As an example, say we have data in ``my_data.csv`` and a model description for DPVI with automatic modelling in ``my_model.txt``. We want 1000 samples of generated data to be stored in ``my_twin.csv`` and fix twinify's internal randomness with a seed for reproducibility. This is how we run twinify::

    twinify vi my_data.csv my_model.txt my_twin --seed=123 --num_synthetic=1000


In the case that we wrote a model with NumPyro instead of relying on twinify's automatic modelling, our call would like like::


    twinify vi my_data.csv my_numpyro_model.py my_twin --seed=123 --num_synthetic=1000


Assuming that the data is entirely categorical and that we have set up a list of marginal queries in ``my_queries.txt``, we can run twinify using NAPSU-MQ with the following command::


    twinify napsu my_data.csv my_queries.txt my_twin --seed=123 --num_synthetic=1000
