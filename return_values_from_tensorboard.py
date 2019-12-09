import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def return_tensorborad_data(path, scalar_name):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_loss =   event_acc.Scalars(scalar_name)
    validation_loss = event_acc.Scalars('val_'+scalar_name)

    steps = len(training_loss)
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in list(range(steps)):
        y[i, 0] = training_loss[i][2] # value
        y[i, 1] = validation_loss[i][2]

    return x,y
