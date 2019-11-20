from my_lib import *
from metrics import *
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Which slices in which patient contains tumor
with open('patients_slices.json', 'r') as f:
    slices = json.loads(f.read())

base_path = ''
# Important if one wants more than one class that they are in relative order ['t1', 't1ce', 't2', 'flair']
modalities = ['t1']
#lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1]
lrs = [1e-4]
input_size = (176, 176, len(modalities))

# Loading of patients
X_train, Y_train, labels_train = load_patients_new_again(i=0, j=234, modalities=modalities, slices=slices, base_path=base_path)
X_val, Y_val, labels_val = load_patients_new_again(i=235, j=284, modalities=modalities, slices=slices, base_path=base_path)
X_test, Y_test, labels_test = load_patients_new_again(i=285, j=334, modalities=modalities, slices=slices, base_path=base_path)

for lr in lrs:
    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + modalities[0] + '-lr-' + str(lr) + "-weights-{epoch:02d}-{val_dice:.2f}.hdf5"
    log_dir = "logs/run1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + modalities[0] + '_lr_' + str(lr)

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_dice', mode='max', verbose=1, period=10)
    #es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir, histogram_freq=1)

    metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
    unet = unet_dong_et_al(input_size=input_size, num_classes=4, lr=lr, drop_rate=0.2,
                                      metrics=metrics, loss=gen_dice_loss)

    X_train = X_train.reshape(-1, 176, 176, 1)
    Y_train = Y_train.reshape(-1, 176, 176, 4)
    validation_data = (X_val.reshape(-1, 176, 176, 1), Y_val.reshape(-1, 176, 176, 4))

    history = unet.fit(x=X_train,
                       y=Y_train,
                       batch_size=128,
                       epochs=100,
                       verbose=1,
                       #callbacks=[cp, es, tbc],
                       callbacks=[cp, tbc],
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
