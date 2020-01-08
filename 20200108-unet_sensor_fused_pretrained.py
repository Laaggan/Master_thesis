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

train_ind, val_ind, _ = create_train_test_split()

batch_size = 16
lr = 1e-4
epochs = 100
total_num_slices = 1.5e4
seed = 1
num_batches_in_epoch = int(total_num_slices // batch_size)

# Initialize and load weights of pretrained networks
input_size_small = (176, 176, 1)
unet_t1 = unet_dong_each_mod(input_size=input_size_small, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)
unet_t1ce = unet_dong_each_mod(input_size=input_size_small, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)
unet_t2 = unet_dong_each_mod(input_size=input_size_small, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)
unet_flair = unet_dong_each_mod(input_size=input_size_small, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)

t1_weights = '20200107-separately_trained_weights_sensor_fusion/20200107-102802-t1-lr-0.0001-n-79-dong_small.hdf5'
t1ce_weights = "20200107-separately_trained_weights_sensor_fusion/20200107-102838-t1ce-lr-0.0001-n-79-dong_small.hdf5"
t2_weights = "20200107-separately_trained_weights_sensor_fusion/20200107-102913-t2-lr-0.0001-n-79-dong_small.hdf5"
flair_weights = "20200107-separately_trained_weights_sensor_fusion/20200107-102947-flair-lr-0.0001-n-79-dong_small.hdf5"

unet_t1.load_weights(t1_weights)
unet_t1ce.load_weights(t1ce_weights)
unet_t2.load_weights(t2_weights)
unet_flair.load_weights(flair_weights)

# Setup the model
unet = sensor_fused_unet_v3(unet_t1, unet_t1ce, unet_t2, unet_flair, lr=lr, loss='categorical_crossentropy', metrics=metrics)

X_train, Y_train = load_patients_numpy("data_numpy_separate_patients_original_size", train_ind, cropping=True)
X_val, Y_val = load_patients_numpy("data_numpy_separate_patients_original_size", val_ind, cropping=True)

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
weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-all-lr-' + str(lr)\
               + '-n-' + str(X_train.shape[0]) + "-weights_data_aug_fit_gen.hdf5"
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-all" \
          + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0]) + '_data_aug_fit_gen'

cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', mode='auto', verbose=1, period=1)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
tbc = TensorBoard(log_dir=log_dir)

tot_generator = zip(input_generator, label_generator)

unet.fit_generator(
        tot_generator,
        steps_per_epoch=num_batches_in_epoch,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=[cp, es, tbc]
)
