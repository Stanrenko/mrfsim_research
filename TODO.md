# TODO

README Update
Need to modify neurite package post install for voxelmorph to work (https://github.com/tensorflow/tensorflow/issues/70796)

In envs/mrfsim-research/lib/python3.10/site-packages/neurite/tf/losses.py and envs/mrfsim-research/lib/python3.10/site-packages/neurite/tf/metrics.py

Replace:

from tensorflow.keras.losses import mean_absolute_error as l1
from tensorflow.keras.losses import mean_squared_error as l2

By:

from tensorflow.keras.losses import MAE as l1
from tensorflow.keras.losses import MSE as l2
