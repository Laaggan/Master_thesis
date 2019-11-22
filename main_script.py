from my_lib import *
from metrics import *
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

modalities = ['t1ce']
lrs = [1e-4]
input_size = (176, 176, len(modalities))

X, Y = load_slices_from_numpy_test()

# Splitting into training and testing
num_slices = X.shape[0]
#ind1 = int(np.floor(num_slices*0.7))
ind2 = int(np.floor(num_slices*(0.7+0.15)))
#ind3 = int(num_slices - 1)
X_train = X[0:ind2, :, :, 1]
#X_test = X[(ind2+1):ind3, :, :, :]
Y_train = Y[0:ind2, :, :, :]
#Y_test = Y[(ind2+1):ind3, :, :, :]
del X, Y

for lr in lrs:
    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + modalities[0] + '-lr-' + str(lr) + "-weights-{epoch:02d}-{val_dice:.2f}.hdf5"
    log_dir = "logs/numpy_data_dong_et_al/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + modalities[0] + '_lr_' + str(lr)

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_dice', mode='max', verbose=1, period=10)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir, histogram_freq=1)

    metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
    unet = unet_dong_et_al(input_size=input_size, num_classes=4, lr=lr, drop_rate=0.2, metrics=metrics,
                           loss='categorical_crossentropy')

    X_train = X_train.reshape(-1, 176, 176, 1)
    Y_train = Y_train.reshape(-1, 176, 176, 4)
    #validation_data = (X_val.reshape(-1, 176, 176, 1), Y_val.reshape(-1, 176, 176, 4))

    history = unet.fit(x=X_train,
                       y=Y_train,
                       batch_size=16,
                       epochs=100,
                       verbose=1,
                       callbacks=[cp, es, tbc],
                       validation_split=0.21,
                       #validation_data=validation_data,
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
