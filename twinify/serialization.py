from typing import Tuple, Any, Union, BinaryIO
import numpy as np
import jax.numpy as jnp
import jax

ENDIANESS = 'big'
STR_ENCODING = 'utf8'
STRLEN_BYTES = 2
ITEMCOUNT_BYTES = 4
ARRLEN_BYTES = 8

def write_shape(shape_tuple: Tuple[int], writer: BinaryIO) -> None:
    num_vals = len(shape_tuple)
    writer.write(num_vals.to_bytes(ITEMCOUNT_BYTES, ENDIANESS))
    for val in shape_tuple:
        writer.write(val.to_bytes(ITEMCOUNT_BYTES, ENDIANESS))

def read_shape(reader: BinaryIO) -> Tuple[int]:
    num_vals = int.from_bytes(reader.read(ITEMCOUNT_BYTES), ENDIANESS, signed=False)
    return tuple(
        int.from_bytes(reader.read(ITEMCOUNT_BYTES), ENDIANESS, signed=False) for _ in range(num_vals)
    )

def write_dtype(dtype: np.dtype, writer: BinaryIO) -> None:
    dtype_str = str(dtype).encode(STR_ENCODING)
    dtype_str_len = len(dtype_str)
    writer.write(dtype_str_len.to_bytes(STRLEN_BYTES, ENDIANESS))
    writer.write(dtype_str)

def read_dtype(reader: BinaryIO) -> np.dtype:
    dtype_str_len = int.from_bytes(reader.read(STRLEN_BYTES), ENDIANESS, signed=False)
    dtype_str = reader.read(dtype_str_len).decode(STR_ENCODING)
    return np.dtype(dtype_str)

def write_array(arr: Union[np.ndarray, jnp.ndarray], writer: BinaryIO) -> None:
    shape = np.shape(arr)
    dtype = arr.dtype
    write_shape(shape, writer)
    write_dtype(dtype, writer)
    arr_bytes = arr.tobytes()
    arr_bytes_len = len(arr_bytes)
    writer.write(arr_bytes_len.to_bytes(ARRLEN_BYTES, ENDIANESS))
    writer.write(arr_bytes)

def read_array(reader: BinaryIO) -> np.ndarray:
    shape = read_shape(reader)
    dtype = read_dtype(reader)
    arr_bytes_len = int.from_bytes(reader.read(ARRLEN_BYTES), ENDIANESS, signed=False)
    arr_bytes = reader.read(arr_bytes_len)
    return np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)

def write_params(params, writer: BinaryIO) -> jax.tree_util.PyTreeDef:
    flat_data, treedef = jax.tree_flatten(params)
    num_data_sites = len(flat_data)
    writer.write(num_data_sites.to_bytes(ITEMCOUNT_BYTES, ENDIANESS))

    for data_site in flat_data:
        if not isinstance(data_site, (np.ndarray, jnp.ndarray)):
            data_site = np.array(data_site)

        write_array(data_site, writer)

    return treedef

def read_params(reader: BinaryIO, treedef: jax.tree_util.PyTreeDef) -> Any:
    num_data_sites = int.from_bytes(reader.read(ITEMCOUNT_BYTES), ENDIANESS, signed=False)
    flat_data = [read_array(reader) for _ in range(num_data_sites)]
    return jax.tree_unflatten(treedef, flat_data)
