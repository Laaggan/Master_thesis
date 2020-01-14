from my_lib import *
from old_functionality import *
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

batch_size = 16
train_ind, val_ind = create_train_test_split()

X_train, Y_train = load_patients_numpy(path_to_folder='data_numpy_separate_patients_original_size', indices=train_ind, cropping=True)
X_val, Y_val = load_patients_numpy(path_to_folder='data_numpy_separate_patients_original_size', indices=val_ind, cropping=True)

H, W = X_train.shape[1], X_train.shape[2]
input_size = (H, W, 4)
lr = 1e-4

# Where to save logs and weights
weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-all-lr-' + str(lr)\
               + '-n-' + str(X_train.shape[0]) + "-weights_he_normal_l2_0.001.hdf5"
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-all" \
          + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0]) + '_he_normal_l2_0.001'

cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', mode='auto', verbose=1, period=1)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
tbc = TensorBoard(log_dir=log_dir)

metrics = [dice, dice_en_metric, dice_core_metric, dice_whole_metric, 'accuracy']
unet = unet_dong_et_al2(input_size=input_size, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)

X_train = X_train.reshape(-1, H, W, 4)
Y_train = Y_train.reshape(-1, H, W, 4)
validation_data = (X_val.reshape(-1, H, W, 4), Y_val.reshape(-1, H, W, 4))

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