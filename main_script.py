from my_lib import *
from metrics import *
from keras import initializers
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

modalities = ['t1ce']
lrs = [1e-4]
H, W = 176, 176
input_size = (H, W, 4)
n = 2
ind = np.arange(0, n)
X_train, Y_train = load_patients_numpy(path_to_folder='/data_numpy/data_numpy_separate_patients_original_size', indices=ind, cropping=True)

m = 2
ind_val = np.arange(n, n+m)
X_val, Y_val = load_patients_numpy(path_to_folder='/data_numpy/data_numpy_separate_patients_original_size', indices=ind_val, cropping=True)

for lr in lrs:
    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + modalities[0] + '-lr-' + str(lr)\
                   + '-n-' + str(X_train.shape[0]) + "-weights-{epoch:02d}-{val_dice:.2f}.hdf5"
    log_dir = "logs/numpy_data_dong_et_al/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + modalities[0] \
              + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0])

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_dice', mode='max', verbose=1, period=10)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir, histogram_freq=1)

    metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
    unet = lee_unet(input_size=input_size, num_classes=4, lr=1e-4, loss='categorical_crossentropy', metrics=metrics)

    X_train = X_train.reshape(-1, H, W, 1)
    Y_train = Y_train.reshape(-1, H, W, 4)
    validation_data = (X_val.reshape(-1, 176, 176, 1), Y_val.reshape(-1, 176, 176, 4))

    history = unet.fit(x=X_train,
                       y=Y_train,
                       batch_size=16,
                       epochs=100,
                       verbose=1,
                       callbacks=[cp, es, tbc],
                       validation_data=validation_data,
                       shuffle=True,
                       class_weight=None,
                       sample_weight=None,
                       initial_epoch=0,
                       steps_per_epoch=None,
                       validation_steps=None,
                       validation_freq=1)

    # Might as well save this object
    with open('HistoryDict_' + modalities[0] + '_' + str(lr), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
