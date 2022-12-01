
import importlib.util
import inspect
from re import A
import traceback
import pandas as pd
import numpy as np
from typing import Any, Callable, Tuple, Iterable, Dict, Union, Optional, Sequence
import twinify.dpvi
from twinify.dpvi.modelling import automodel
from numpyro.infer.autoguide import AutoDiagonalNormal
from twinify import DataDescription
import os
import argparse

__all__ = ['load_custom_numpyro_model', 'ModelException', 'NumpyroModelParsingUnknownException', 'NumpyroModelParsingException']

TPreprocessFunction = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]
TGuardedPreprocessFunction = Callable[[pd.DataFrame], pd.DataFrame]
TPostprocessFunction = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]
TGuardedPostprocessFunction = Callable[[pd.DataFrame], pd.DataFrame]
TModelFunction = twinify.dpvi.ModelFunction
TGuideFunction = twinify.dpvi.GuideFunction
TModelFactoryFunction = Callable[[argparse.Namespace, Iterable[str], DataDescription], Union[TModelFunction, Tuple[TModelFunction, TGuideFunction]]]


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


def guard_preprocess(preprocess_fn: TPreprocessFunction) -> TGuardedPreprocessFunction:
    def wrapped_preprocess(train_df: pd.DataFrame) -> pd.DataFrame:
        try:
            train_data = preprocess_fn(train_df)
        except TypeError as e:
            if str(e).find('positional argument') != -1:
                raise ModelException("FAILED DURING PREPROCESSING DATA", "Custom preprocessing functions must accept a single pandas.DataFrame as argument.")
            else:
                raise ModelException("FAILED DURING PREPROCESSING DATA", base_exception=e) from e
        except Exception as e:
            raise ModelException("FAILED DURING PREPROCESSING DATA", base_exception=e) from e

        if not isinstance(train_data, (pd.DataFrame, pd.Series)):
            raise ModelException("FAILED DURING PREPROCESSING DATA", f"Custom preprocessing functions must return a pandas.DataFrame or pandas.Series, but returned a {type(train_data)} instead.")

        if isinstance(train_data, pd.Series):
            train_data = train_data.to_frame()

        return train_data

    return wrapped_preprocess

@guard_preprocess
def default_preprocess(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    return train_df

def guard_postprocess(postprocess_fn: TPostprocessFunction) -> TGuardedPostprocessFunction:
    def wrapped_postprocess(posterior_samples: pd.DataFrame) -> pd.DataFrame:
        try:
            retval = postprocess_fn(posterior_samples)
        except TypeError as e:
            if str(e).find('positional argument') != -1:
                raise ModelException("FAILED DURING POSTPROCESSING DATA", "Custom postprocessing functions must accept a pandas.DataFrame as argument.")
            else:
                raise e
        except Exception as e:
            raise ModelException("FAILED DURING POSTPROCESSING DATA", base_exception=e) from e

        if not isinstance(retval, (pd.DataFrame, pd.Series)):
            raise ModelException("FAILED DURING POSTPROCESSING DATA", f"Custom postprocessing functions must return a single pandas.DataFrame (or Series); got {type(retval)}.")

        if isinstance(retval, pd.Series):
            retval = retval.to_frame()
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
    ) -> Tuple[TModelFunction, TGuideFunction, TGuardedPreprocessFunction, TGuardedPostprocessFunction]:

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    try:
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
    except Exception as e: # handling errors in py-file parsing
        raise NumpyroModelParsingException("Unable to read the specified file as a Python module.", e) from e

    # load the model function from the module
    model = None
    guide = None
    try: model = model_module.model
    except AttributeError:
        # model file did not directly contain a model function; check if it has model_factory
        try:
            model_factory = model_module.model_factory
        except AttributeError:
            raise NumpyroModelParsingException("Model module does neither specify a 'model' nor a 'model_factory' function.")
        try:
            model_factory_return = model_factory(args, unknown_args, orig_data)
        except TypeError as e:
            if str(e).find('positional argument') != -1:
                raise ModelException("FAILED IN MODEL FACTORY", f"Custom model_factory functions must accept a namespace of parsed arguments, an iterable of unparsed arguments and a pandas.DataFrame as arguments.")
            raise e
        except Exception as e:
            raise ModelException('FAILED IN MODEL FACTORY', base_exception=e) from e

        # determine whether model_factory returned model function or (model, guide) tuple
        if (type(model_factory_return) is tuple
                and isinstance(model_factory_return[0], TModelFunction)
                and isinstance(model_factory_return[1], TGuideFunction)):
            model, guide = model_factory_return
        elif isinstance(model_factory_return, TModelFunction):
            model = model_factory_return
        else:
            raise ModelException('FAILED IN MODEL FACTORY', f"Custom model_factory functions must return either a model function or a tuple consisting of model and guide function, but returned {type(model_factory_return)}.")
    except Exception as e: raise NumpyroModelParsingUnknownException('model', e) from e

    if not isinstance(model, Callable):
        raise NumpyroModelParsingException(f"'model' must be a function; got {type(model)}")
    model = guard_model(model)

    if guide is None:
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
