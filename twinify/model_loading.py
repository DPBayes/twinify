
import importlib.util
import inspect
import traceback
import pandas as pd
import numpy as np
from typing import Callable, Tuple, Iterable, Dict, Union, Optional
from twinify import automodel
from numpyro.infer.autoguide import AutoDiagonalNormal

__all__ = ['load_custom_numpyro_model', 'NumpyroModelParsingUnknownException', 'NumpyroModelParsingException']

TPreprocessFunction = Callable[[pd.DataFrame], Tuple[Iterable[pd.DataFrame], int]]
TPostprocessFunction = Callable[[Dict[str, np.ndarray], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]
TOldPostprocessFunction = Callable[[pd.DataFrame], pd.DataFrame]
TModelFunction = Callable
TGuideFunction = Callable


def preprocess_wrapper(preprocess_fn: TPreprocessFunction) -> TPreprocessFunction:
    def wrapped_preprocess(train_df):
        train_data, num_data = preprocess_fn(train_df)
        if isinstance(train_data, pd.DataFrame):
            train_data = (train_data,)

        if not isinstance(train_data, Iterable):
            print(f"ERROR: Custom preprocessing functions must return an (interable of) pandas.DataFrame as first returned value, but returned a {type(train_data)} instead.")
            exit(4)

        return train_data, num_data

    return wrapped_preprocess

@preprocess_wrapper
def default_preprocess(train_df: pd.DataFrame):
    return train_df, len(train_df)

def postprocess_wrapper(postprocess_fn: Union[TPostprocessFunction, TOldPostprocessFunction]) -> TPostprocessFunction:
    num_parameters = len(inspect.signature(postprocess_fn).parameters)
    if num_parameters == 1:
        print("WARNING: Custom postprocessing function using old signature, which is deprecated!")
        def wrapped_postprocess_old(posterior_samples: Dict[str, np.ndarray], orig_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            syn_data = pd.DataFrame(posterior_samples['x'], columns = orig_df.columns)
            encoded_syn_data = postprocess_fn(syn_data) # TODO: handle potential errors
            return encoded_syn_data, syn_data
        return wrapped_postprocess_old
    else:
        def wrapped_postprocess(posterior_samples: Dict[str, np.ndarray], orig_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            return postprocess_fn(posterior_samples, orig_df) # TODO: handle potential errors

        return wrapped_postprocess

class NumpyroModelParsingException(Exception):

    def __init__(self, msg: str, base_exception: Optional[Exception]=None) -> None:
        self.msg = msg
        self.base = base_exception

        full_message = f"\n#### FAILED TO PARSE THE MODEL SPECIFICATION ####\n##   {msg}"

        super().__init__(full_message)

class NumpyroModelParsingUnknownException(NumpyroModelParsingException):

    def __init__(self, function_name: str, base_exception: Exception) -> None:
        self.__init__(f"Unknown error while trying to access function '{function_name}' from model module.", base_exception)

def load_custom_numpyro_model(model_path: str) -> Tuple[TModelFunction, TGuideFunction, TPreprocessFunction, TPostprocessFunction]:
    try:
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
    except Exception as e: # handling errors in py-file parsing
        raise NumpyroModelParsingException("Unable to read the specified file as a Python module.", e) from e

    # load the model function from the module
    try: model = model_module.model
    except AttributeError: raise NumpyroModelParsingException("Model module does not specify a 'model' function.")
    except Exception as e: raise NumpyroModelParsingUnknownException('model', e) from e

    try: guide = model_module.guide
    except AttributeError: guide = AutoDiagonalNormal(model)
    except Exception as e: raise NumpyroModelParsingUnknownException('guide', e) from e

    # try to obtain preprocessing function from custom model
    try: preprocess_fn = preprocess_wrapper(model_module.preprocess)
    except AttributeError: preprocess_fn = default_preprocess
    except Exception as e: raise NumpyroModelParsingUnknownException('preprocess', e) from e

    # try to obtain postprocessing function from custom model
    try: postprocess_fn = postprocess_wrapper(model_module.postprocess)
    except AttributeError:
        print("Warning: Your model does not specify a postprocessing function for generated samples.")
        print("     Using default, which assumes that your model only produces samples at sample site 'x' and outputs samples as they are.")
        postprocess_fn = automodel.postprocess_function_factory([])
    except Exception as e: raise NumpyroModelParsingUnknownException('postprocess', e) from e

    return model, guide, preprocess_fn, postprocess_fn