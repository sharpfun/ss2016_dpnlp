import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,nvcc.flags=-D_FORCE_INLINES,floatX=float32"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu"


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


source_path = 'dataset/shakespeare.hdf5'


with h5py.File(source_path) as f:
    vocab_size = len(json.loads(f.attrs['index_to_char']))
    instances_num = f['x'].shape[0]


train_dataset = H5PYDataset(source_path, which_sets=('train',))


hidden_layer_dim = 300

x = tensor.lmatrix('x')
y = tensor.lmatrix('y')

lookup_input = LookupTable(
    name='lookup_input',
    length=vocab_size,
    dim=hidden_layer_dim,
    weights_init=initialization.Uniform(width=0.01),
    biases_init=Constant(0))
lookup_input.initialize()

rnn = SimpleRecurrent(
    name='hidden',
    dim=hidden_layer_dim,
    activation=Tanh(),
    weights_init=initialization.Uniform(width=0.01))
rnn.initialize()

linear_output = Linear(
    name='linear_output',
    input_dim=hidden_layer_dim,
    output_dim=vocab_size,
    weights_init=initialization.Uniform(width=0.01),
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
    Checkpoint(path="./checkpoint.zip", every_n_batches=100)
]


from blocks.model import Model

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=DataStream.default_stream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size=1000)
    ),
    model=Model(y_est),
    extensions=extensions
)

main_loop.run()


