from my_lib import *
from metrics import *
import os
from keras.utils import plot_model
from cube_slider import cube_show_slider
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_movie(patient, trained_network, file_name):
    X, Y = load_patients_numpy('data_numpy_separate_patients_original_size', indices=[patient], cropping=True)

    Yhat = np.zeros(Y.shape)
    for i, x in enumerate(X):
        input_ = x.reshape((1, 176, 176, 4))
        yhat = trained_network.predict(input_)
        Yhat[i, :, :, :] = yhat.squeeze()

    predictions = OHE_uncoding(Yhat, mapping)
    ground_truth = OHE_uncoding(Y, mapping)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fps = 15

    # First set up the figure, the axis, and the plot element we want to animate
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    a = ground_truth[20, :, :]
    im1 = ax1.imshow(a)
    im2 = ax2.imshow(a)

    ax1.set_title('Ground truth')
    ax2.set_title('Predictions')

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        im1.set_array(ground_truth[i, :, :])
        im2.set_array(predictions[i, :, :])
        return [im1, im2]

    anim = animation.FuncAnimation(
                                   fig,
                                   animate_func,
                                   frames = ground_truth.shape[0],
                                   interval = 1000 / fps, # in ms
                                   )

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(file_name + '.mp4', writer=writer)

    print('Done!')