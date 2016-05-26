import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

from blocks.bricks.recurrent import BaseRecurrent, Initializable, recurrent, SimpleRecurrent
from blocks.roles import add_role, WEIGHT
from blocks.bricks.base import lazy, application
from blocks.utils import shared_floatx_nans
from theano import tensor
from blocks.bricks import Activation
from fuel.datasets.hdf5 import h5py, H5PYDataset
from fuel.datasets import IndexableDataset
from collections import Mapping, OrderedDict
import numbers


class CocoDataset(H5PYDataset):
    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                if isinstance(request, numbers.Integral):
                    data[i] = data[i].reshape(shapes[i])
                else:
                    for j in range(len(data[i])):
                        data[i][j] = data[i][j].reshape(shapes[i][j])
        data[1] = data[1].T
        return tuple(data)

import os.path

if os.path.isfile('/projects/korpora/mscoco/coco.hdf5'):
    train_dataset = CocoDataset('/projects/korpora/mscoco/coco.hdf5', which_sets=('train',))
else:
    train_dataset = CocoDataset('/home/kroman/Desktop/cogsys/ss2016/pm1 dlnlp/4/solution/dataset/coco.hdf5', which_sets=('train',))

import json
if os.path.isfile('/projects/korpora/mscoco/coco/cocotalk.json'):
    words_dic = json.loads(open('/projects/korpora/mscoco/coco/cocotalk.json').read())["ix_to_word"]
else:
    words_dic = json.loads(open('/home/kroman/Desktop/cogsys/ss2016/pm1 dlnlp/4/solution/dataset/coco/cocotalk.json').read())["ix_to_word"]


print "num_examples", train_dataset.num_examples


class ContextSimpleRecurrent(SimpleRecurrent):
    """very simple recurrent that's context-aware"""

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, mask=None, context=None):
        """Same as SimpleRecurrent.apply except with an additional argument:

        context : :class:`~tensor.TensorVariable`
            Not actually used here, but needed for readout to take it into account

        I was afraid of calling super().apply in this case because I didn't know how that would
        work with all the decorators.
        """
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    def get_dim(self, name):
        if name == "context":
            # this is a dirty hack! It relies on you specifying a keyword arg to Merge
            # See below for details
            return 1000
        return super(ContextSimpleRecurrent, self).get_dim(name)


from blocks.bricks import Tanh
from blocks.bricks.sequence_generators import SequenceGenerator, Readout, LookupFeedback, SoftmaxEmitter
from blocks import initialization
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Adam


context_dim = 100

# length of training sentence
sequence_dim = 16

words_count = len(words_dic)+1 # dictionary key starts from 1


generator = SequenceGenerator(
    readout=Readout(
        readout_dim=words_count,
        source_names=["states", "context"],
        name='readout',
        feedback_brick=LookupFeedback(
            num_outputs=words_count,
            feedback_dim=words_count),
        emitter=SoftmaxEmitter()
    ),
    transition=ContextSimpleRecurrent(
        name='image',
        activation=Tanh(),
        dim=context_dim,
        weights_init=initialization.IsotropicGaussian(0.1)
    ),
    weights_init=initialization.IsotropicGaussian(0.1),
    biases_init=initialization.Constant(0.0))

generator.initialize()

x = tensor.matrix('image')
y = tensor.lmatrix('sequence')

cost = generator.cost(y, context=x)

cg = ComputationGraph([cost])

algorithm = GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=Adam()
)


from blocks.extensions import Timing, FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.model import Model

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream.default_stream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, 1)
    ),
    model=Model(cost),
    extensions=[
        Timing(),
        FinishAfter(after_n_epochs=20),
        TrainingDataMonitoring(
            variables=[cost],
            prefix="train",
            after_epoch=True
        ),
        Printing(),
        ProgressBar(),
        Checkpoint("checkpoint.tar")
    ]
)

main_loop.run()
