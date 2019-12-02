from my_lib import *
from metrics import *
from keras import initializers
import datetime
import pickle
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

modalities = {
    't1': 0,
    't1ce': 1,
    't2': 2,
    'flair': 3
}
# Seems to be a fine learning rate
lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
batch_size = 16
# There is 335 patients in total. -> indices [0, 334]
n = 335
np.random.seed(42)
ind = np.arange(n)
np.random.shuffle(ind)
ind1 = int(np.ceil(len(ind) * 0.7))
ind2 = int(np.ceil(len(ind) * 0.85))

train_ind = ind[0:ind1]
val_ind = ind[ind1:ind2]

train_ind = [0]
val_ind = [0]

X_train, Y_train = load_patients_numpy(path_to_folder='data_numpy_separate_patients_original_size', indices=train_ind, cropping=True)
X_val, Y_val = load_patients_numpy(path_to_folder='data_numpy_separate_patients_original_size', indices=val_ind, cropping=True)

H, W = X_train.shape[1], X_train.shape[2]
input_size = (H, W, 4)

for lr in lrs:
    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-all-lr-' + str(lr)\
                   + '-n-' + str(X_train.shape[0]) + "-weights.hdf5"
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-all" \
              + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0])

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_dice', mode='max', verbose=1, period=1)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir)

    metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
    unet = lee_unet2(input_size=input_size, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)

    X_train = X_train.reshape(-1, H, W, 4)
    Y_train = Y_train.reshape(-1, H, W, 4)
    validation_data = (X_val.reshape(-1, H, W, 4), Y_val.reshape(-1, H, W, 4))

    history = unet.fit(x=X_train,
                       y=Y_train,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       callbacks=[tbc, cp, es],
                       validation_data=validation_data,
                       shuffle=True,
                       class_weight=None,
                       sample_weight=None,
                       initial_epoch=0,
                       steps_per_epoch=None,
                       validation_steps=None)