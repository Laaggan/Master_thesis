from my_lib import *
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Which slices in which patient contains tumor
with open('patients_slices.json', 'r') as f:
    slices = json.loads(f.read())

base_path = ''
# Important if one wants more than one class that they are in relative order ['t1', 't1ce', 't2', 'flair']
modalities = ['t1']
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1]
input_size = (176, 176, len(modalities))

# Loading of patients
X_train, Y_train, labels_train = load_patients_new_again(i=0, j=234, modalities=modalities, slices=slices, base_path=base_path)
X_val, Y_val, labels_val = load_patients_new_again(i=235, j=284, modalities=modalities, slices=slices, base_path=base_path)
X_test, Y_test, labels_test = load_patients_new_again(i=285, j=334, modalities=modalities, slices=slices, base_path=base_path)

for lr in lrs:
    # Where to save logs and weights
    weights_path = modalities[0] + '-lr-' + str(lr) + "-weights-{epoch:02d}-{val_mean_iou:.2f}.hdf5"
    log_dir = "logs/run1/" + modalities[0] + '_lr_' + str(lr) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_mean_iou', mode='max', verbose=1, period=1)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir, histogram_freq=1)

    my_unet = unet(input_size=input_size, num_classes=4, learning_rate=lr, drop_rate=0.2, metrics=[mean_iou, dice_coef])

    Y_train = Y_train.reshape(Y_train.shape[0], -1, 4)
    validation_data = (X_val, Y_val.reshape(Y_val.shape[0], -1, 4))

    history = my_unet.fit(x=X_train,
                          y=Y_train,
                          batch_size=64,
                          epochs=200,
                          verbose=1,
                          callbacks=[cp, es, tbc],
                          validation_split=0.0,
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
