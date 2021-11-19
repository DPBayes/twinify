
import importlib.util
import inspect
from re import A
import traceback
import pandas as pd
import numpy as np
from typing import Any, Callable, Tuple, Iterable, Dict, Union, Optional, Sequence
from twinify import automodel
from numpyro.infer.autoguide import AutoDiagonalNormal
import os
import argparse

__all__ = ['load_custom_numpyro_model', 'ModelException', 'NumpyroModelParsingUnknownException', 'NumpyroModelParsingException']

TPreprocessFunction = Callable[[pd.DataFrame], Tuple[Iterable[pd.DataFrame], int]]
TWrappedPreprocessFunction = Callable[[pd.DataFrame], Tuple[Iterable[pd.DataFrame], int, Sequence[str]]]
TWrappedPostprocessFunction = Callable[[Dict[str, np.ndarray], pd.DataFrame, Sequence[str]], Tuple[pd.DataFrame, pd.DataFrame]]
TPostprocessFunction = Callable[[Dict[str, np.ndarray], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]
TOldPostprocessFunction = Callable[[pd.DataFrame], pd.DataFrame]
TModelFunction = Callable
TGuideFunction = Callable
TModelFactoryFunction = Callable[[argparse.Namespace, Iterable[str], pd.DataFrame], TModelFunction]


class ModelException(Exception):

    @staticmethod
    def filter_traceback(tb: Iterable[traceback.StackSummary], model_file_path: str) -> Iterable[traceback.StackSummary]:
        assert model_file_path is not None
        model_file_name = os.path.basename(model_file_path)
        encountered_model = False
        for frame in tb:
            if frame.filename.endswith(model_file_name):
                encountered_model = True
            if encountered_model:
                yield frame

    def __init__(self, title: str, msg: str=None, base_exception: Optional[Exception]=None) -> None:
        self.msg = msg if msg is not None else "Uncategorised error"
        self.base = base_exception
        self.title = title

        super().__init__(self.format_message())

    def format_message(self, model_file_path: Optional[str]=None) -> str:
        full_message = f"\n#### {self.title.upper()} ####\n##   {self.msg}"
        if self.base is not None:
            full_message += f"\nTechnical error description below:\n"
            tb = traceback.extract_tb(self.base.__traceback__)
            if model_file_path is not None:
                tb = self.filter_traceback(tb, model_file_path)
            full_message += "\n".join(traceback.format_list(tb))
            full_message += "\n".join(traceback.format_exception_only(type(self.base), self.base))

        return full_message

class NumpyroModelParsingException(ModelException):

    def __init__(self, msg: str, base_exception: Optional[Exception]=None) -> None:
        self.msg = msg
        self.base = base_exception

        title = "FAILED TO PARSE THE MODEL SPECIFICATION"

        super().__init__(title, msg, base_exception)

class NumpyroModelParsingUnknownException(NumpyroModelParsingException):

    def __init__(self, function_name: str, base_exception: Exception) -> None:
        self.__init__(f"Uncategorised error while trying to access function '{function_name}' from model module.", base_exception)


def guard_preprocess(preprocess_fn: TPreprocessFunction) -> TWrappedPreprocessFunction:
    def wrapped_preprocess(train_df):
        try:
            retval = preprocess_fn(train_df)
        except TypeError as e:
            if str(e).find('positional argument') != -1:
                raise ModelException("FAILED DURING PREPROCESSING DATA", "Custom preprocessing functions must accept a single pandas.DataFrame as argument.")
            else:
                raise ModelException("FAILED DURING PREPROCESSING DATA", base_exception=e) from e
        except Exception as e:
            raise ModelException("FAILED DURING PREPROCESSING DATA", base_exception=e) from e

        if isinstance(retval, (pd.DataFrame, pd.Series)):
            train_data = (retval,)
            num_data = len(retval)
        else:
            try:
                train_data, num_data = retval
            except:
                raise ModelException("FAILED DURING PREPROCESSING DATA", "Custom preprocessing functions must return (train_data, num_data), where train_data is a tuple of pandas.DataFrame or pandas.Series and num_data is the number of data points.")
            if isinstance(train_data, (pd.DataFrame, pd.Series)):
                train_data = (train_data,)

        if not isinstance(train_data, Iterable):
            raise ModelException("FAILED DURING PREPROCESSING DATA", f"Custom preprocessing functions must return an (iterable of) pandas.DataFrame or pandas.Series as first returned value, but returned a {type(train_data)} instead.")

        all_feature_names = []
        for df in train_data:
            if isinstance(df, pd.DataFrame):
                all_feature_names += list(df.columns)
            elif isinstance(df, pd.Series):
                all_feature_names.append(df.name)
            else:
                raise ModelException("FAILED DURING PREPROCESSING DATA", f"Custom preprocessing functions must return an (iterable of) pandas.DataFrame or pandas.Series as first returned value, but at least one {type(df)} was returned.")


        return train_data, num_data, all_feature_names

    return wrapped_preprocess


@guard_preprocess
def default_preprocess(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    return train_df, len(train_df)

def guard_postprocess(postprocess_fn: Union[TPostprocessFunction, TOldPostprocessFunction]) -> TWrappedPostprocessFunction:
    num_parameters = len(inspect.signature(postprocess_fn).parameters)
    if num_parameters == 1:
        print("WARNING: Custom postprocessing function using old signature, which is deprecated!")
        def wrapped_postprocess_old(posterior_samples: Dict[str, np.ndarray], orig_df: pd.DataFrame, feature_names: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
            try:
                if 'x' not in posterior_samples:
                    raise ModelException("FAILED DURING POSTPROCESSING DATA", f"For the specified postprocessing function with a single argument, the 'model'  function must combine all features at sample site 'x'.")
                syn_data = posterior_samples['x']
                syn_data = np.squeeze(syn_data, 1)  # Predictive produces (num_data, 1, num_features)
                if len(syn_data.shape) == 1: syn_data = np.expand_dims(syn_data, -1)  # in case num_features = 1 results in () sample shape
                if syn_data.shape[-1] != len(feature_names):
                    raise ModelException("FAILED DURING POSTPROCESSING DATA", f"Number of features in synthetic data ({syn_data.shape[-1]}) does not match number of features used in training ({len(feature_names)}).")
                syn_data = pd.DataFrame(syn_data, columns = feature_names)
                encoded_syn_data = postprocess_fn(syn_data)
                if not isinstance(encoded_syn_data, pd.DataFrame):
                    raise ModelException("FAILED DURING POSTPROCESSING DATA", f"Custom postprocessing functions must return a pd.DataFrame, got {type(encoded_syn_data)}")
                return syn_data, encoded_syn_data
            except ModelException as e: raise e
            except Exception as e: raise ModelException("FAILED DURING POSTPROCESSING DATA", base_exception=e) from e
        return wrapped_postprocess_old
    else:
        def wrapped_postprocess(posterior_samples: Dict[str, np.ndarray], orig_df: pd.DataFrame, feature_names: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
            try:
                # numpyro.Predictive produces (num_data, 1, num_features) for samples in plate (because batch size seen by model is 1 there)
                # -> squeeze out intermediate 1
                posterior_samples = {
                    site: np.squeeze(samples, 1) for site, samples in posterior_samples.items()
                }
                retval = postprocess_fn(posterior_samples, orig_df)
            except TypeError as e:
                if str(e).find('positional argument') != -1:
                    raise ModelException("FAILED DURING POSTPROCESSING DATA", "Custom postprocessing functions must accept two single pandas.DataFrame as arguments.")
                else:
                    raise e
            except Exception as e:
                raise ModelException("FAILED DURING POSTPROCESSING DATA", base_exception=e) from e
            if not isinstance(retval, tuple) or len(retval) != 2 or not isinstance(retval[0], pd.DataFrame) or not isinstance(retval[1], pd.DataFrame):
                got_msg = f"({', '.join(type(x) for x in retval)})" if isinstance(retval, tuple) else type(retval)
                raise ModelException("FAILED DURING POSTPROCESSING DATA", f"Custom postprocessing functions must return (syn_data, encoded_syn_data), each being a pd.DataFrame; got {got_msg}.")
            return retval
        return wrapped_postprocess

def guard_model(model_fn: TModelFunction) -> TModelFunction:
    def wrapped_model(*args, **kwargs) -> Any:
        try:
            return model_fn(*args, **kwargs)
        except TypeError as e:
            if str(e).find('positional argument') != -1 or str(e).find('num_obs_total') != -1:
                raise ModelException("FAILED IN MODEL", f"Custom model functions can accept any number of positional arguments for training but must default all of them to None for synthesising data. They also must accept the num_obs_total keyword argument.", base_exception=e) from e
            else:
                raise ModelException("FAILED IN MODEL", base_exception=e) from e
        except Exception as e:
            raise ModelException("FAILED IN MODEL", base_exception=e) from e
    return wrapped_model

def load_custom_numpyro_model(
        model_path: str, args: argparse.Namespace, unknown_args: Iterable[str], orig_data: pd.DataFrame
    ) -> Tuple[TModelFunction, TGuideFunction, TWrappedPreprocessFunction, TWrappedPostprocessFunction]:

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    try:
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
    except Exception as e: # handling errors in py-file parsing
        raise NumpyroModelParsingException("Unable to read the specified file as a Python module.", e) from e

    # load the model function from the module
    try: model = model_module.model
    except AttributeError:
        try:
            model_factory = model_module.model_factory
        except AttributeError:
            raise NumpyroModelParsingException("Model module does neither specify a 'model' nor a 'model_factory' function.")
        try:
            model = model_factory(args, unknown_args, orig_data)
        except TypeError as e:
            if str(e).find('positional argument') != -1:
                raise ModelException("FAILED IN MODEL FACTORY", f"Custom model_factory functions must accept a namespace of parsed arguments, an iterable of unparsed arguments and a pandas.DataFrame as arguments.")
            raise e
        except Exception as e:
            raise ModelException('FAILED IN MODEL FACTORY', base_exception=e) from e
    except Exception as e: raise NumpyroModelParsingUnknownException('model', e) from e

    if not isinstance(model, Callable):
        raise NumpyroModelParsingException(f"'model' must be a function; got {type(model)}")
    model = guard_model(model)

    try: guide = model_module.guide
    except AttributeError: guide = AutoDiagonalNormal(model)
    except Exception as e: raise NumpyroModelParsingUnknownException('guide', e) from e

    # try to obtain preprocessing function from custom model
    try: preprocess_fn = guard_preprocess(model_module.preprocess)
    except AttributeError: preprocess_fn = default_preprocess
    except Exception as e: raise NumpyroModelParsingUnknownException('preprocess', e) from e

    # try to obtain postprocessing function from custom model
    try: postprocess_fn = guard_postprocess(model_module.postprocess)
    except AttributeError:
        print("Warning: Your model does not specify a postprocessing function for generated samples.")
        print("     Using default, which assumes that your model only produces samples at sample site 'x' and outputs samples as they are.")
        postprocess_fn = automodel.postprocess_function_factory([])
    except Exception as e: raise NumpyroModelParsingUnknownException('postprocess', e) from e

    return model, guide, preprocess_fn, postprocess_fn
