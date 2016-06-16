import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

from theano import tensor
from blocks import initialization
from blocks.initialization import Constant
from blocks.bricks import Tanh
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.bricks import Linear, Softmax, NDimensionalSoftmax
from fuel.datasets.hdf5 import H5PYDataset
import h5py
import json
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp


source_path = 'dataset/shakespeare.hdf5'


with h5py.File(source_path) as f:
    charset_size = len(json.loads(f.attrs['index_to_char']))
    instances_num = f['x'].shape[0]


class MyDataset(H5PYDataset):
    def get_data(self, state=None, request=None):
        data = list(super(MyDataset, self).get_data(state, request))
        data[0] = data[0].T
        data[1] = data[1].T
        return tuple(data)


train_dataset = MyDataset(source_path, which_sets=('train',))


hidden_layer_dim = 500

x = tensor.lmatrix('x')
y = tensor.lmatrix('y')

lookup_input = LookupTable(
    name='lookup_input',
    length=charset_size,
    dim=hidden_layer_dim,
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
lookup_input.initialize()

linear_input = Linear(
    name='linear_input',
    input_dim=hidden_layer_dim,
    output_dim=hidden_layer_dim,
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
linear_input.initialize()

rnn = SimpleRecurrent(
    name='hidden',
    dim=hidden_layer_dim,
    activation=Tanh(),
    weights_init=initialization.Uniform(width=0.01))
rnn.initialize()

linear_output = Linear(
    name='linear_output',
    input_dim=hidden_layer_dim,
    output_dim=charset_size,
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
linear_output.initialize()

softmax = NDimensionalSoftmax(name='ndim_softmax')

activation_input = lookup_input.apply(x)
hidden = rnn.apply(linear_input.apply(activation_input))
activation_output = linear_output.apply(hidden)
y_est = softmax.apply(activation_output, extra_ndim=1)

cost = softmax.categorical_cross_entropy(y, activation_output, extra_ndim=1).mean()


from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Adam

cg = ComputationGraph([cost])

step_rules = [RMSProp(learning_rate=0.002, decay_rate=0.95), StepClipping(1.0)]


algorithm = GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=CompositeRule(step_rules)
)


from blocks.extensions import Timing, FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.extensions.saveload import Checkpoint


from blocks.model import Model

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream.default_stream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size=200)
    ),
    model=Model(y_est),
    extensions=[
        Timing(),
        FinishAfter(after_n_epochs=300),
        TrainingDataMonitoring(
            variables=[cost],
            prefix="train",
            after_epoch=True
        ),
        Printing(),
        ProgressBar(),
        Checkpoint(path="./checkpoint.zip")
    ]
)

main_loop.run()


