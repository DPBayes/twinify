from jax.config import config
config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)

import jax.numpy as np

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from dppp.minibatch import q_to_batch_size, batch_size_to_q
from dppp.dputil import approximate_sigma_remove_relation
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal

import fourier_accountant

from twinify.infer import train_model, train_model_no_dp, InferenceException
import twinify.automodel as automodel

import numpy as onp

import pandas as pd

import importlib.util

import jax, argparse, pickle


data_path = "./tds_example/tds_all.csv"
model_path = "./crash_cand.txt"
seed = 0
k = 5
preprocess = 1
clipping_threshold = 10000000.0
drop_na = 0
num_epochs = 200
sampling_ratio = 0.01
dp_sigma = .0

onp.random.seed(seed)

# read data
df = pd.read_csv(data_path)

print("Parsing model from txt file (was unable to read it python module containing numpyro code)")
# read model file
model_handle = open(model_path, 'r')
model_str = "".join(model_handle.readlines())
model_handle.close()
features = automodel.parse_model(model_str)
feature_names = [feature.name for feature in features]

# pick features from data according to model file
train_df = df[feature_names]
if drop_na:
	train_df = train_df.dropna()

# NOTE add missing values to features
for feature in features:
	if onp.any(train_df[feature.name].isna()):
		feature._missing_values = True


# data preprocessing: determines number of categories for Categorical
#	distribution and maps categorical values in the data to ints
for feature in features:
	train_df = feature.preprocess_data(train_df)

# build model
model = automodel.make_model(features, k)
model_args_map = automodel.model_args_map

# build variational guide for optimization
guide = AutoDiagonalNormal(make_observed_model(model, model_args_map))

# pick features from data according to model file
num_data = train_df.shape[0]
print("After removing missing values, the data has {} entries with {} features".format(*train_df.shape))

# learn posterior distributions
#posterior_params = train_model(
#	 jax.random.PRNGKey(seed),
#	 model, automodel.model_args_map, guide, None,
#	 train_df.to_numpy(),
#	 batch_size=int(sampling_ratio*len(train_df)),
#	 num_epochs=num_epochs,
#	 dp_scale=dp_sigma,
#	 clipping_threshold=clipping_threshold
#)


#posterior_params = train_model_no_dp(
#	 jax.random.PRNGKey(seed),
#	 model, automodel.model_args_map, guide, None,
#	 train_df.to_numpy(),
#	 batch_size=int(sampling_ratio*len(train_df)),
#	 num_epochs=num_epochs
#)

#
#assert(0==1)

#####################################

from numpyro.optim import Adam
from dppp.modelling import make_observed_model
from numpyro.infer import SVI
from dppp.svi import DPSVI
from numpyro.contrib.autoguide import AutoContinuousELBO
optimizer = Adam(1e-3)
model_args_map = automodel.model_args_map
data = train_df.to_numpy()

#if model_args_map is not None:
#	model = make_observed_model(model, model_args_map)

#svi = SVI(
#	model, guide,
#	optimizer, AutoContinuousELBO(),
#	num_obs_total=data.shape[0]
#)

svi = DPSVI(
	model, guide, optimizer, AutoContinuousELBO(),
	num_obs_total=data.shape[0], clipping_threshold=clipping_threshold,
	dp_scale=dp_sigma,
	map_model_args_fn=model_args_map
)

#######################
from dppp.minibatch import minibatch, subsample_batchify_data
from numpyro.infer.svi import SVIState
rng = jax.random.PRNGKey(0)
batch_size = int(sampling_ratio*len(data))

rng, svi_rng, init_batch_rng = jax.random.split(rng, 3)

init_batching, get_batch = subsample_batchify_data((data,), batch_size)
_, batchify_state = init_batching(init_batch_rng)

batch = get_batch(0, batchify_state)
svi_state = svi.init(svi_rng, *batch)

##### hotfix to prevent double jit compilation #####
###### remove this once fixed in numpyro/jax #######
optim_state = svi_state.optim_state
optim_state = (np.array(svi_state.optim_state[0]), *(optim_state[1:]))
svi_state = SVIState(optim_state, svi_state.rng_key)
init_svi_state = svi_state

rng, epochs_rng = jax.random.split(rng)

batch_0, = get_batch(0, init_batching(jax.random.fold_in(epochs_rng, 0))[1])
svi_state, iter_loss = svi.update(svi_state, batch_0)

batch_1, = get_batch(0, init_batching(jax.random.fold_in(epochs_rng, 1))[1])
svi_state_1, losses_pxg, gradients = svi._compute_per_example_gradients(svi_state, batch_1)

last_state, last_loss = svi.update(svi_state, batch_1)
assert(0==1)
####################################################


def loop_it(start, stop, fn, init_val, do_jit=True):
	if do_jit:
		return jax.lax.fori_loop(start, stop, fn, init_val)
	else:
		#with jax.disable_jit():
		#	val = init_val
		#	for i in range(start, stop):
		#		val = fn(i, val)
		#	return val
		val = init_val
		for i in range(start, stop):
			val = fn(i, val)
		return val


#@jax.jit ## this commented seems to work
def train_epoch(num_epoch_iter, svi_state, batchify_state):
	@jax.jit
	def train_iteration(i, state_and_loss):
		svi_state, loss = state_and_loss
		batch_x, = get_batch(i, batchify_state)
		svi_state, iter_loss = svi.update(svi_state, batch_x)
		return (svi_state, iter_loss)

	#return loop_it(0, 1, train_iteration, (svi_state, 0.), do_jit=True)
	return loop_it(0, 1, train_iteration, (svi_state, 0.), do_jit=False)

def train_iteration(svi_state, batchify_state, i):
	batch_x, = get_batch(i, batchify_state)
	svi_state, iter_loss = svi.update(svi_state, batch_x)
	return svi_state, iter_loss

rng, epochs_rng = jax.random.split(rng)

for i in range(2):
	previous_state = svi_state
	#if i%10==0:
	#	batchify_rng = jax.random.fold_in(epochs_rng, i // 10)
	#	num_batches, batchify_state = init_batching(batchify_rng)
	batchify_rng = jax.random.fold_in(epochs_rng, i)
	num_batches, batchify_state = init_batching(batchify_rng)

	svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
	#svi_state, loss = train_iteration(svi_state, batchify_state, 0)
	if np.isnan(loss):
		raise InferenceException
	loss /= data.shape[0]
	print("epoch {}: loss {}".format(i, loss))


"""
Before lunch
So everything was working with SVI, both the infrastructured version as well as the butchered one
The DP infrastructured fails, and also the butchered fails under certain conditions:
	Testing with 30 epochs, each epoch with 1 iteration
	* If we use train_epoch with jit compilation >> FAIL after 
		(epoch 0: loss 76.14749295229008, epoch 1: loss 70.37675013692096)
	* Without jit compilation and do_jit=False >> NO FAIL
		(epoch 0: loss 76.14749295229008, epoch 1: loss 70.37675013692096)
	* If we use train_iteration >> NO FAIL
		(epoch 0: loss 76.14749295229008, epoch 1: loss 70.37675013692096)
"""
