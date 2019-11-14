from my_lib import *
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Which slices in which patient contains tumor
with open('patients_slices.json', 'r') as f:
    slices = json.loads(f.read())

base_path = ''
# Important if one wants all classes that they are in order ['t1', 't1ce', 't2', 'flair']
modalities = ['t1']
input_size = (176, 176, len(modalities))

# Loading of patients
X_train, Y_train, labels_train = load_patients_new_again(i=0, j=1, modalities=modalities, slices=slices, base_path=base_path)
X_val, Y_val, labels_val = load_patients_new_again(i=301, j=302, modalities=modalities, slices=slices, base_path=base_path)
#X_test, Y_test, labels_test = load_patients_new_again(i=285, j=334, modalities=modalities, slices=slices, base_path=base_path)

# Where to save logs and weights
# fixme: val_acc below should be changed to the chosen metric later
weights_path = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
log_dir = "logs/unet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#cp = ModelCheckpoint(weights_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
cp = ModelCheckpoint(weights_path, verbose=1, period=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
tbc = TensorBoard(log_dir=log_dir, histogram_freq=1)

# fixme: accuracy should be changed to the chosen metric when such exists
my_unet = unet(input_size=input_size, num_classes=4, learning_rate=0.01, drop_rate=0.2, metrics=['accuracy'])

Y_train = Y_train.reshape(Y_train.shape[0], -1, 4)
validation_data = (X_val, Y_val.reshape(Y_val.shape[0], -1, 4))

history = my_unet.fit(x=X_train,
                      y=Y_train,
                      batch_size=16,
                      epochs=1,
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
with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
