# API Overview

Using twinify as a library, you retain full control over data loading, pre- and postprocessing, in contrast to the command line tool.
The main actors in the twinify APIs are `twinify.InferenceModel` and `twinify.InferenceResult`.

## `InferenceModel`
`InferenceModel` fully encapsulates a model and algorithm to fit it to the data. It defines a single function

```
fit(data: pd.DataFrame, rng: d3p.random.PRNGState, epsilon: float, delta: float, **kwargs) -> InferenceResult
```
which takes an input data set given as a [pandas](https://pandas.pydata.org/) DataFrame as well as privacy parameters and a randomness state. It returns
a representation of the model fitted to the data in the form of a `InferenceResult` object.

Currently twinify provides `twinify.dpvi.DPVIModel` and `twinify.napsu_mq.NapsuMQModel` as concrete implementations, with the following initializers:
- `DPVIModel(model: NumPyroModelFunction, guide: Optional[NumPyroGuideFunction] = None, clipping_threshold: float = 1., num_epochs: int = 1000, subsample_ratio: float = 0.01)`
- `NapsuMQModel(column_feature_set: Iterable[FrozenSet[str]], use_laplace_approximation: bool = True)`

## `InferenceResult`
`InferenceResult` represents a learned model from which synthetic data can be generated. To that end it defines the method

```
generate(
        rng: d3p.random.PRNGState,
        num_parameter_samples: int,
        num_data_per_parameter_sample: int = 1,
        single_dataframe: bool = True
    ) -> Union[Iterable[pd.DataFrame], pd.DataFrame]
```

This method first draws `num_parameter_samples` parameter samples from the model posterior represented by the `InferenceResult` object and then
samples `num_data_per_parameter_sample` data points for each parameter sample from the model, and returns them as either one large combined DataFrame or an iterable over one DataFrame per parameter sample.

`InferenceResult` classes also allow saving and loading of learned models via the `save` and static `load` methods respectively.

Note that `DPVIResult.load` requires the same NumPyro model as used for inference to be provided during model loading.
