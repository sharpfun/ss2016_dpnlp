import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,nvcc.flags=-D_FORCE_INLINES,floatX=float32"

from blocks.extensions.saveload import load
from theano import function
import pandas
import numpy


with open('dataset/Aird-v1.abc.txt') as f:
    CORPUS = f.read()#[:100000]


(indices, indexed_letters) = pandas.factorize(list(CORPUS))


main_loop = load('./checkpoint.zip')

model = main_loop.model

print [x.name for x in model.shared_variables]

print len([x for x in model.shared_variables if x.name == "initial_state"])

tensor_initial1 = [x for x in model.shared_variables if x.name == "initial_state"][2]
#tensor_initial2 = [x for x in model.shared_variables if x.name == "initial_state"][1]
#tensor_initial3 = [x for x in model.shared_variables if x.name == "initial_state"][0]
tensor_hidden1_states = [x for x in model.intermediary_variables if x.name == "hidden1_apply_states"][0]
#tensor_hidden2_states = [x for x in model.intermediary_variables if x.name == "hidden2_apply_states"][0]
#tensor_hidden3_states = [x for x in model.intermediary_variables if x.name == "hidden3_apply_states"][0]
tensor_x = [x for x in model.variables if x.name == "x"][0]
tensor_y = [x for x in model.variables if x.name == "ndim_softmax_apply_output"][0]

predict_fun = function([tensor_x], tensor_y, updates=[
    (tensor_initial1, tensor_hidden1_states[0][0]),
    #(tensor_initial2, tensor_hidden2_states[0][0]),
    #(tensor_initial3, tensor_hidden3_states[0][0])
])

predictions = [0]
import time
numpy.random.seed(int(time.time()))
for i in range(500):
    input_char = numpy.zeros((1, 1), dtype=numpy.int32)
    input_char[0][0] = predictions[i]
    predictions.append(numpy.random.choice(len(indexed_letters), 1, p=predict_fun(input_char)[0])[0])


print "Predict:"
print "".join([indexed_letters[x] for x in predictions])