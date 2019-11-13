from my_lib import *
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

base_path = ''

# Loading of patients
train_data = load_patients(i=0, j=1, base_path=base_path)
val_data = load_patients(i=2, j=3, base_path=base_path)

X_train = train_data[0]
Y_train = train_data[1]

X_val = val_data[0]
Y_val = val_data[1]

filepath="weights-{epoch:02d}-{val_IoU:.2f}.hdf5"
cp = ModelCheckpoint(filepath, monitor='val_IoU', verbose=1, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

log_dir="logs/unet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbc = TensorBoard(log_dir=log_dir, histogram_freq=1)

my_unet = unet(input_size=(240, 240, 1), num_classes=2, metrics=[dice_coefficient, IoU])

Y_train = Y_train.reshape(Y_train.shape[0], -1, 2)
validation_data = (X_val, Y_val.reshape(Y_val.shape[0], -1, 2))

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

# I guess this is unnecessary when you save tensorboard files
#with open('history.json', 'w') as f:
#    f.write(json.dumps(history))