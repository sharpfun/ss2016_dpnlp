import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

from blocks.extensions.saveload import load
from theano import function
import numpy
import h5py
import json


source_path = 'dataset/shakespeare.hdf5'


with h5py.File(source_path) as f:
    vocab = json.loads(f.attrs['index_to_char'])
    vocab_size = len(vocab)
    instances_num = f['x'].shape[0]


main_loop = load('./checkpoint.zip')

model = main_loop.model

print [x.name for x in model.shared_variables]

print [x.name for x in model.variables]

tensor_initial = [x for x in model.shared_variables if x.name == "initial_state"][0]
tensor_hidden_states = [x for x in model.intermediary_variables if x.name == "hidden_apply_states"][0]
tensor_x = [x for x in model.variables if x.name == "x"][0]
tensor_y = [x for x in model.variables if x.name == "ndim_softmax_apply_output"][0]

predict_fun = function([tensor_x], tensor_y, updates=[
    (tensor_initial, tensor_hidden_states[0][0]),
])

predictions = [0]
import time
numpy.random.seed(int(time.time()))
for i in range(500):
    input_char = numpy.zeros((1, 1), dtype=numpy.int32)
    input_char[0][0] = predictions[i]
    predictions.append(numpy.random.choice(vocab_size, 1, p=predict_fun(input_char)[0])[0])


print "Predict:"
print "".join([vocab[x] for x in predictions])