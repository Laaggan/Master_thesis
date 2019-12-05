from my_lib import *
from metrics import *
from keras import initializers
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras

input_size = (176, 176, 4)
metrics = [dice, dice_en_metric, dice_core_metric, dice_whole_metric, 'accuracy']

train_ind, val_ind = create_train_test_split()

batch_size=16
lr=1e-4
epochs = 100
total_num_slices = 1.5e4
num_batches_in_epoch = int(total_num_slices // batch_size)

# Setup the model
unet = unet_dong_et_al2(input_size=input_size, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)
train_choice = np.random.choice(train_ind, len(val_ind))

X_train, Y_train = load_patients_numpy("data_numpy_separate_patients_original_size", train_choice, cropping=True)
X_val, Y_val = load_patients_numpy("data_numpy_separate_patients_original_size", val_ind, cropping=True)

im_gen = ImageDataGenerator(
   rotation_range=20,
   horizontal_flip=True,
   vertical_flip=True,
   width_shift_range=0.1,
   height_shift_range=0.1,
   shear_range=2,
   zoom_range=0.1)

import sys

def create_a_batch(aug_iter, batch_size):
    aug_data = [next(aug_iter)[0].astype(np.float32) for i in range(batch_size)]
    X = np.zeros((batch_size, 176, 176, 4))
    Y = np.zeros((batch_size, 176, 176, 4))

    for i in range(batch_size):
        X[i, :, :, :] = aug_data[i][:, :, 0:4].reshape((1, 176, 176, 4))
        Y[i, :, :, :] = aug_data[i][:, :, 4:8].reshape((1, 176, 176, 4))
    Y = np.round(Y)
    return X, Y

training_results = []
validation_results = []
patients_in_aug_iter = 10
num_batches = int((patients_in_aug_iter*66)//batch_size)

for epoch in range(epochs):
    print("Epoch", epoch)
    for _ in range(int(len(train_ind)//patients_in_aug_iter)):
        curr_ind = np.random.choice(train_ind, patients_in_aug_iter)
        print("Patients:", curr_ind)
        X_aug, Y_aug = load_patients_numpy("data_numpy_separate_patients_original_size", curr_ind, cropping=True)
        data = np.concatenate((X_aug, Y_aug), axis=3)
        aug_iter = im_gen.flow(data)
        for batch_id in range(num_batches):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            i = round(((batch_id+1)*batch_size))
            sys.stdout.write("Sample " + str(i))
            sys.stdout.flush()

            X_batch, Y_batch = create_a_batch(aug_iter, batch_size)
            loss = unet.train_on_batch(X_batch, Y_batch)

    train_eval = unet.evaluate(X_train, Y_train)
    #[dice, dice_en_metric, dice_core_metric, dice_whole_metric, 'accuracy']
    s_train = "Train; Loss:{}, DSC:{}, DSC enhancing:{}, DSC core:{}, DSC whole:{}, accuracy:{}"
    # [dice, dice_en_metric, dice_core_metric, dice_whole_metric, 'accuracy']
    print(s_train.format(*train_eval))
    training_results.append(train_eval)

    val_eval = unet.evaluate(X_val, Y_val)
    s_val = "Validation; Loss:{}, DSC:{}, DSC enhancing:{}, DSC core:{}, DSC whole:{}, accuracy:{}"
    # [dice, dice_en_metric, dice_core_metric, dice_whole_metric, 'accuracy']
    print(s_val.format(*val_eval))
    validation_results.append(val_eval)

with open(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'train_results.pkl', 'wb') as f:
    pickle.dump(training_results, f)

with open(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'validation_results.pkl', 'wb') as f:
    pickle.dump(validation_results, f)

weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-all-lr-' + str(lr)\
               + '-n-' + str(X_train.shape[0]) + "-weights_he_normal_l2_0.001-data_augmentation.hdf5"

unet.save_weights(weights_path)