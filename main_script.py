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
lr = 1e-4
batch_size = 16
num_modalities = 1
# There is 335 patients in total. -> indices [0, 334]
n = 335
np.random.seed(42)
ind = np.arange(n)
np.random.shuffle(ind)
ind1 = int(np.ceil(len(ind) * 0.7))
ind2 = int(np.ceil(len(ind) * 0.85))

train_ind = ind[0:ind1]
val_ind = ind[ind1:ind2]

X_train_raw, Y_train = load_patients_numpy(path_to_folder='data_numpy_separate_patients_original_size', indices=train_ind, cropping=True)
X_val_raw, Y_val = load_patients_numpy(path_to_folder='data_numpy_separate_patients_original_size', indices=val_ind, cropping=True)

H, W = X_train_raw.shape[1], X_train_raw.shape[2]
input_size = (H, W, num_modalities)

for mod in modalities:
    # Extract modality of interest
    i = modalities[mod]
    X_train = X_train_raw[:, :, :, i]
    X_val = X_val_raw[:, :, :, i]

    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + mod + '-lr-' + str(lr)\
                   + '-n-' + str(X_train.shape[0]) + "-weights.hdf5"
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + mod \
              + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0])

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_dice', mode='max', verbose=1, period=1)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir)

    metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
    unet = lee_unet2(input_size=input_size, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)

    X_train = X_train.reshape(-1, H, W, num_modalities)
    Y_train = Y_train.reshape(-1, H, W, 4)
    validation_data = (X_val.reshape(-1, H, W, num_modalities), Y_val.reshape(-1, H, W, 4))

    history = unet.fit(x=X_train,
                       y=Y_train,
                       batch_size=batch_size,
                       epochs=100,
                       verbose=1,
                       callbacks=[tbc, cp, es],
                       validation_data=validation_data,
                       shuffle=True,
                       class_weight=None,
                       sample_weight=None,
                       initial_epoch=0,
                       steps_per_epoch=None,
                       validation_steps=None)