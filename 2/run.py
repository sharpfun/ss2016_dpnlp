from nltk.corpus import brown
from collections import Counter
import numpy

from fuel.datasets import Dataset
import re

numpy.set_printoptions(threshold=numpy.nan)


class Words2VecDataSet(Dataset):
    def __init__(self, features, targets, words_bag_size, bag_words):
        self.provides_sources = ["features", "targets"]
        self.features = features
        self.targets = targets
        self.axis_labels = None
        self.words_bag_size = words_bag_size
        self.bag_words = bag_words

        super(Words2VecDataSet, self).__init__()

    def num_instances(self):
        return len(self.targets)

    @staticmethod
    def from_words(source_words, window=1):
        words_occurence = Counter(source_words)

        freq_words = [word for (word, occurence) in words_occurence.items() if occurence > 5 and re.search("[a-zA-Z]", word)]

        print "Amount of words:", len(freq_words)

        word_to_id = {}
        bag_words = []
        counter = 0
        for word in freq_words:
            word_to_id[word] = counter
            bag_words.append(word)

            counter += 1

        # will store indexes of words, not vector
        features = []
        targets = []

        words_len = len(source_words)

        for i in range(words_len):
            if source_words[i] not in word_to_id:
                continue
            features_of_one = []
            for j in range(i-window, i+window+1):
                if j == i or j < 0 or j == words_len:
                    continue
                if source_words[j] in word_to_id:
                    features_of_one.append(word_to_id[source_words[j]])
            if features_of_one:
                targets.append(word_to_id[source_words[i]])
                features.append(numpy.array(features_of_one, dtype=numpy.int))

        words_bag_size = len(freq_words)

        return Words2VecDataSet(features, targets, words_bag_size, bag_words)

    def get_data(self, state=None, request=None):
        return [self.features[req_id] for req_id in request], [self.targets[req_id] for req_id in request]


train_dataset = Words2VecDataSet.from_words(brown.words()[:10000])


from blocks.bricks import Linear, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.initialization import Uniform, Constant
from theano import tensor
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.bricks.cost import MisclassificationRate


x = tensor.lmatrix('features')
y = tensor.ivector('targets')

hidden_layer_size = 200

layer1 = LookupTable(name='layer1',
                     length=train_dataset.words_bag_size,
                     dim=hidden_layer_size,
                     weights_init=Uniform(mean=0, std=0.01),
                     biases_init=Constant(0))
act1_mean = tensor.mean(layer1.apply(x), axis=1)
layer2 = Linear(name='layer2',
                input_dim=layer1.output_dim,
                output_dim=train_dataset.words_bag_size,
                weights_init=Uniform(mean=0, std=0.01),
                biases_init=Constant(0))
act2_softmax = Softmax().apply(layer2.apply(act1_mean))

layer1.initialize()
layer2.initialize()

missclass = MisclassificationRate().apply(y, act2_softmax)

cost = CategoricalCrossEntropy().apply(y, act2_softmax)

cg = ComputationGraph([cost])

W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)

cost = cost + 0.00001 * (W1**2).sum() + 0.00005 * (W2**2).sum()

cost.name = 'cost'

from blocks.algorithms import GradientDescent, Scale

algorithm = GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=Scale(learning_rate=0.1)
)

from blocks.extensions import Timing, FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop


from blocks.extensions import SimpleExtension


class SaveWeights(SimpleExtension):
    def __init__(self, layers, prefixes, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(SaveWeights, self).__init__(**kwargs)
        self.step = 1
        self.layers = layers
        self.prefixes = prefixes

    def do(self, callback_name, *args):
        for i in xrange(len(self.layers)):
            filename = "%s_%d.npy" % (self.prefixes[i], self.step)
            numpy.save(filename, self.layers[i].parameters[0].get_value())
        self.step += 1


extensions = [
    Timing(),
    FinishAfter(after_n_epochs=20),
    TrainingDataMonitoring(
        variables=[cost, missclass],
        prefix="train",
        after_epoch=True
    ),
    Printing(),
    ProgressBar(),
    SaveWeights(layers=[layer1, layer2], prefixes=["layer1", "layer2"])
]

print "num_instances", train_dataset.num_instances()


from blocks.model import Model

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream.default_stream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_instances(), 1)
    ),
    model=Model(cost),
    extensions=extensions
)

main_loop.run()

from tsne import *
import matplotlib.pyplot as plt

W1 = numpy.load("layer1_20.npy")

Y = tsne(W1, 2, 50, 20.0)

fig, ax = plt.subplots()
ax.scatter(Y[:,0], Y[:,1])

for i, word in enumerate(train_dataset.bag_words):
    x,y = Y[i]
    ax.annotate(word, (x,y))

plt.show()
