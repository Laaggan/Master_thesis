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
train_ind = train_ind[0:1]
val_ind = val_ind[0:1]

batch_size = 4
lr = 1e-4
epochs = 100
total_num_slices = 1.5e4
seed = 1
num_batches_in_epoch = int(total_num_slices // batch_size)

def sensor_fused_unet(input_size, lr, metrics, num_classes):
    kernel_size = 3
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2)
    }
    num_channels = input_size[-1]
    input = Input(shape=input_size)

    branch_outputs = []
    for i in range(num_channels):
        # Slicing the ith channel:
        in_ = Lambda(lambda x: x[:, :, :, i:(i+1)])(input)

        # Setting up your per-channel layers (replace with actual sub-models):
        conv1 = Conv2D(64, (3, 3), input_shape=(176, 176, 1), **conv_kwargs)(in_)
        pool1 = MaxPooling2D(**pooling_kwargs)(conv1)

        conv2 = Conv2D(128, (3, 3), **conv_kwargs)(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), **conv_kwargs)(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), **conv_kwargs)(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)

        up6 = Conv2DTranspose(512, (2, 2), **conv_transpose_kwargs)(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        up7 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(merge6)
        merge7 = concatenate([conv3, up7], axis=3)

        up8 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(merge7)
        merge8 = concatenate([conv2, up8], axis=3)

        up9 = Conv2DTranspose(64, (2, 2), **conv_transpose_kwargs)(merge8)
        merge9 = concatenate([conv1, up9], axis=3)

        branch_outputs.append(merge9)

    # Concatenating together the per-channel results:
    out = Concatenate()(branch_outputs)

    # Adding some further layers (replace or remove with your architecture):
    out = Conv2D(128, (3, 3), **conv_kwargs)(out)
    out = Conv2D(128, (3, 3), **conv_kwargs)(out)
    out = Conv2D(num_classes, (1, 1), **conv_kwargs)(out)
    out = Activation('softmax')(out)

    # Building model:
    model = Model(inputs=input, outputs=out)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=metrics)
    return model

# Setup the model
unet = sensor_fused_unet(input_size=input_size, num_classes=4, lr=lr, metrics=metrics)

X_train, Y_train = load_patients_numpy("data_numpy_separate_patients_original_size", train_ind[0:], cropping=True)
X_val, Y_val = load_patients_numpy("data_numpy_separate_patients_original_size", val_ind[0:], cropping=True)

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
               + '-n-' + str(X_train.shape[0]) + "-new_sensor_fusion.hdf5"
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-all" \
          + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0]) + 'new_sensor_fusion'

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
