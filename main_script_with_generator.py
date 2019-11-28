from my_lib import *
from metrics import *
from keras import initializers
import datetime
import pickle
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data_generator import Generator

modalities = {
    't1': 0,
    't1ce': 1,
    't2': 2,
    'flair': 3
}
# Seems to be a fine learning rate
lr = 1e-4
num_modalities = 1
# There is 335 patients in total. -> indices [0, 334]
num_patients = 335
np.random.seed(42)
ind = np.arange(num_patients)
np.random.shuffle(ind)
ind1 = int(np.ceil(len(ind) * 0.7))
ind2 = int(np.ceil(len(ind) * 0.85))

train_ind = ind[0:ind1]
val_ind = ind[ind1:ind2]

path_to_data = 'data_numpy_separate_patients_original_size'
# How many patients will be in each update
# 2 patients will equal a batch size of ~100
batch_size = 2

X_val_raw, Y_val = load_patients_numpy(
    path_to_folder=path_to_data,
    indices=val_ind,
    cropping=True
)

H, W = X_val_raw.shape[1], X_val_raw.shape[2]
input_size = (H, W, num_modalities)

for mod in modalities:
    # Extract modality of interest
    i = modalities[mod]
    X_val = X_val_raw[:, :, :, i]

    training_batch_generator = Generator(
        indices=train_ind,
        path_to_folder=path_to_data,
        modality=i,
        batch_size=batch_size,
        cropping=True
    )

    # Where to save logs and weights
    weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + mod + '-lr-' + str(lr)\
                   + '-weights.hdf5'
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + mod \
              + '-lr-' + str(lr) + '-fit_generator'

    cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_dice', mode='max', verbose=1, period=1)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
    tbc = TensorBoard(log_dir=log_dir)

    metrics = [dice, dice_whole_metric, dice_en_metric, dice_core_metric]
    unet = lee_unet2(input_size=input_size, num_classes=4, lr=lr, loss='categorical_crossentropy', metrics=metrics)

    validation_data = (X_val.reshape(-1, H, W, num_modalities), Y_val.reshape(-1, H, W, 4))

    unet.fit_generator(
        generator=training_batch_generator,
        steps_per_epoch=num_patients // batch_size,
        epochs=100,
        callbacks=[tbc, cp, es],
        verbose=1,
        validation_data=validation_data
    )
