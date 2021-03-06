from my_lib import *
from metrics import *
from keras import initializers
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
import keras.backend as K

modalities = {
    't1': 0,
    't1ce': 1,
    't2': 2,
    'flair': 3
}

input_size = (176, 176, 1)
metrics = [dice, dice_en_metric, dice_core_metric, dice_whole_metric, 'accuracy']
train_ind, val_ind, _ = create_train_test_split()

batch_size = 16
lr = 1e-4
epochs = 100
total_num_slices = 1.5e4
seed = 1
num_batches_in_epoch = int(total_num_slices // batch_size)

for mod in modalities:
    unet = unet_dong_single_mod(input_size=input_size, num_classes=4, lr=lr, loss='categorical_crossentropy',
                                metrics=metrics)
    X_train, Y_train = load_patients_numpy("data_numpy_separate_patients_original_size", train_ind, cropping=True)
    X_val, Y_val = load_patients_numpy("data_numpy_separate_patients_original_size", val_ind, cropping=True)

    ind = modalities[mod]
    X_train = X_train[:, :, :, ind].reshape([-1] + list(input_size))
    X_val = X_val[:, :, :, ind].reshape([-1] + list(input_size))

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1)

    input_generator = train_datagen.flow(
            X_train,
            batch_size=batch_size,
            seed=seed
    )

    label_generator = train_datagen.flow(
            Y_train,
            batch_size=batch_size,
            seed=seed
    )

    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + mod + '-lr-' + str(lr)\
                   + '-n-' + str(X_train.shape[0]) + "-dong_small.hdf5"
    log_dir = "20191209-training_modalities_separately/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + mod \
              + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0]) + '-dong_small'

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', mode='auto', verbose=1, period=1)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir)

    tot_generator = zip(input_generator, label_generator)
    unet.fit_generator(
            tot_generator,
            steps_per_epoch=num_batches_in_epoch,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=[cp, es, tbc])
    K.clear_session()
