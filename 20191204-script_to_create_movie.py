from create_movie_of_results import create_movie
import numpy as np
from metrics import *
from my_lib import *

n = 335
train_ind, val_ind = create_train_test_split()
patient = val_ind[7]

metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
input_size = (176, 176, 4)
unet = lee_unet2(input_size=input_size, num_classes=4, lr=1e-4, loss='categorical_crossentropy',
                 metrics=metrics)

os.chdir("/Users/linuslagergren/Google Drive/EXJOBB/20191128_weights/")
weights_path = "20191128-201206-all-lr-0.0001-n-15539-weights.hdf5"
unet.load_weights(weights_path)
os.chdir("/Users/linuslagergren/Master_thesis/")
file_name = "results_from_function_2"

create_movie(patient, unet, file_name)