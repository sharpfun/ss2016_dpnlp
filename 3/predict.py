from blocks.extensions.saveload import load
from theano import function
import pandas
import numpy


with open('dataset/shakespeare_input.txt') as f:
    CORPUS = "".join(f.readlines())[:20000]


(indices, indexed_letters) = pandas.factorize(list(CORPUS))


main_loop = load('./checkpoint.zip')

model = main_loop.model

tensor_initial = [x for x in model.shared_variables if x.name == "initial_state"][0]
tensor_hidden_states = [x for x in model.intermediary_variables if x.name == "hidden_apply_states"][0]
tensor_x = [x for x in model.variables if x.name == "x"][0]
tensor_y = [x for x in model.variables if x.name == "ndim_softmax_apply_output"][0]

predict_fun = function([tensor_x], tensor_y, updates=[(tensor_initial, tensor_hidden_states[0][0])])

predictions = [0]
for i in range(500):
    input_char = numpy.zeros((1, 1), dtype=numpy.int64)
    input_char[0][0] = predictions[i]
    predictions.append(numpy.random.choice(len(indexed_letters), 1, p=predict_fun(input_char)[0])[0])


print "Predict:"
print "".join([indexed_letters[x] for x in predictions])