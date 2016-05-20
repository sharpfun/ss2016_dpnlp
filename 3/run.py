import numpy
import pandas
from theano import tensor
from blocks import initialization
from blocks.initialization import Constant
from blocks.bricks import Tanh
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Linear, Softmax, NDimensionalSoftmax


with open('dataset/shakespeare_input.txt') as f:
    CORPUS = "".join(f.readlines())[:20000]


(indices, indexed_letters) = pandas.factorize(list(CORPUS))

print indexed_letters

seqlength = 1000
instances_num = len(CORPUS)/seqlength
dimension = len(indexed_letters)
repeat = 30


train_data_x = numpy.zeros((seqlength*repeat, instances_num), dtype=numpy.int64)
train_data_y = numpy.zeros((seqlength*repeat, instances_num), dtype=numpy.int64)


for j in range(instances_num):
    for k in range(repeat):
        for i in range(seqlength-1):
            train_data_x[i+k*seqlength][j] = indices[i+j*seqlength]
            train_data_y[i+k*seqlength][j] = indices[i+j*seqlength+1]


from fuel.datasets import IndexableDataset
from collections import OrderedDict

train_dataset = IndexableDataset(OrderedDict([('x', train_data_x), ('y', train_data_y)]))

hidden_layer_dim = 250

x = tensor.lmatrix('x')
y = tensor.lmatrix('y')

lookup_input = LookupTable(
    name='lookup_input',
    length=dimension,
    dim=hidden_layer_dim,
    weights_init=initialization.IsotropicGaussian(0.01),
    biases_init=Constant(0))
lookup_input.initialize()

rnn = SimpleRecurrent(
    name='hidden',
    dim=hidden_layer_dim,
    activation=Tanh(),
    weights_init=initialization.IsotropicGaussian(0.01))
rnn.initialize()

linear_output = Linear(
    name='linear_output',
    input_dim=hidden_layer_dim,
    output_dim=dimension,
    weights_init=initialization.IsotropicGaussian(0.01),
    biases_init=Constant(0))
linear_output.initialize()

softmax = NDimensionalSoftmax(name='ndim_softmax')

activation_input = lookup_input.apply(x)
hidden = rnn.apply(activation_input)
activation_output = linear_output.apply(hidden)
y_est = softmax.apply(activation_output, extra_ndim=1)

cost = softmax.categorical_cross_entropy(y, activation_output, extra_ndim=1).mean()


from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Adam

cg = ComputationGraph([cost])


algorithm = GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=Adam()
)


from blocks.extensions import Timing, FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.extensions.saveload import Checkpoint


extensions = [
    Timing(),
    FinishAfter(after_n_epochs=500),
    TrainingDataMonitoring(
        variables=[cost],
        prefix="train",
        after_epoch=True
    ),
    Printing(),
    ProgressBar(),
    Checkpoint(path="./checkpoint.zip")
]


from blocks.model import Model

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream.default_stream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(instances_num, batch_size=1)
    ),
    model=Model(y_est),
    extensions=extensions
)

main_loop.run()


