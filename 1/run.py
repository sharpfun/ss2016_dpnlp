import numpy

attributes_to_feature_array = []


with open('dataset/agaricus-lepiota.attributes') as f:
    lines = [x.strip('\n') for x in f.readlines()]
    for line in lines:
        (label, variates_str) = line.split(":")
        variates = [x.split("=") for x in line.split(",")]
        attributes_to_feature_array.append({})
        counter = 0
        for variate in variates:
            one_zeros = numpy.zeros(len(variates), dtype=numpy.int)
            one_zeros[counter] = 1
            counter += 1
            attributes_to_feature_array[-1][variate[1].strip()] = one_zeros

features = []
targets = []

with open('dataset/agaricus-lepiota.data') as f:
    lines = [x.strip('\n').split(",") for x in f.readlines()]

    for line in lines:
        targets.append(attributes_to_feature_array[0][line[0]])

        counter = 1
        one_line_features = []
        for feature in line[1:]:
            one_line_features.append(attributes_to_feature_array[counter][feature])

            counter += 1

        features.append(numpy.concatenate(one_line_features))

features = numpy.array(features, dtype=numpy.float32)
targets = numpy.array(targets, dtype=numpy.int)

print features.shape
print targets.shape

train_size = int(len(targets) * 0.9)

features_train = features[:train_size]
targets_train = targets[:train_size]

features_test = features[train_size:]
targets_test = targets[train_size:]

from fuel.datasets import IndexableDataset
from collections import OrderedDict

train_indexable_dataset = IndexableDataset(
    indexables=OrderedDict([('features', features_train), ('targets', targets_train)])
)

test_indexable_dataset = IndexableDataset(
    indexables=OrderedDict([('features', features_test), ('targets', targets_test)])
)

from blocks.bricks import Linear, Logistic, Softmax
from blocks.initialization import Constant
from theano import tensor
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.bricks.cost import MisclassificationRate


x = tensor.matrix('features')
y = tensor.lmatrix('targets')

lin1 = Linear(name='lin1', input_dim=126, output_dim=50, weights_init=Constant(0.005), biases_init=Constant(0))
act1_sigmoid = Logistic().apply(lin1.apply(x))
lin2 = Linear(name='lin2', input_dim=50, output_dim=2, weights_init=Constant(0.001), biases_init=Constant(0))
act2_softmax = Softmax().apply(lin2.apply(act1_sigmoid))

lin1.initialize()
lin2.initialize()

missclass = MisclassificationRate().apply(y.argmax(axis=1), act2_softmax)
missclass.name = 'missclassification'

cost = CategoricalCrossEntropy().apply(y, act2_softmax)

comp_graph = ComputationGraph([cost])

W1, W2 = VariableFilter(roles=[WEIGHT])(comp_graph.variables)

cost = cost + 0.005 * (W1**2).sum() + 0.005 * (W2**2).sum()
cost.name = 'cost'

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from fuel.transformers import Flatten
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop


extensions = [
    FinishAfter(after_n_epochs=10),
    DataStreamMonitoring(
        variables=[cost, missclass],
        data_stream=Flatten(
            DataStream.default_stream(
                dataset=test_indexable_dataset,
                iteration_scheme=SequentialScheme(test_indexable_dataset.num_examples, 10)
            ),
            which_sources=('features',)
        ),
        prefix="test"
    ),
    Printing(),
    ProgressBar()
]



main_loop = MainLoop(
    extensions=extensions,
    algorithm=GradientDescent(
        cost=cost,
        parameters=comp_graph.parameters,
        step_rule=Scale(learning_rate=0.1)
    ),
    data_stream=DataStream.default_stream(
        dataset=train_indexable_dataset,
        iteration_scheme=SequentialScheme(train_indexable_dataset.num_examples, 10)
    )
)

main_loop.run()

