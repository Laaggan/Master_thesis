from my_lib import *
from metrics import *
import cv2
from keras import initializers
import datetime
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras

input_size = (176, 176, 1)
metrics = [dice, dice_binary_metric, 'accuracy']

batch_size = 2
lr = 1e-4
epochs = 100
total_num_slices = 1.5e4
seed = 1
num_batches_in_epoch = int(total_num_slices // batch_size)
inds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 138, 141, 142, 143, 145, 146, 147, 149, 151, 153, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 207, 208, 210, 212, 214, 215, 217, 218, 219, 220, 221, 222, 223, 224, 226, 227, 228, 229, 230, 231, 232, 233, 235, 236, 237, 239, 240, 241, 242, 243, 244, 245, 246, 247, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 279]

# Setup the model
unet = unet_dong_et_al2(input_size=input_size, num_classes=2, lr=lr, loss='categorical_crossentropy', metrics=metrics)

base_path ='SU-HGG-SkullStripped'

np.random.seed(42)
inds = np.array(inds)
np.random.shuffle(inds)
ind1 = int(np.ceil(len(inds) * 0.7))
ind2 = int(np.ceil(len(inds) * 0.85))
ind3 = len(inds)

train_ind = inds[0:ind1]
val_ind = inds[ind1:ind2]
test_ind = inds[ind2:ind3]

def load_doctor_data(indices):
    data = []
    labels = []
    lens = []
    inds = []

    for i in indices:
        print(i)
        try:
            if i <= 9:
                data_path = "/GBM0" + str(i) + "_SS.nii.gz"
                label_path = "/GBM0" + str(i) + "_label.nii.gz"
            else:
                data_path = "/GBM" + str(i) + "_SS.nii.gz"
                label_path = "/GBM" + str(i) + "_label.nii.gz"

            X = nib.load(base_path + data_path)
            Y = nib.load(base_path + label_path)

            X = X.get_fdata()
            X = zscore_norm(X)
            Y = Y.get_fdata()

            X = cv2.resize(X, (176, 176), interpolation=cv2.INTER_CUBIC)
            Y = cv2.resize(Y, (176, 176), interpolation=0)

            X = np.moveaxis(X, -1, 0)
            Y = np.moveaxis(Y, -1, 0)
            org_shape = Y.shape

            mask = np.sum(Y.reshape(org_shape[0], -1), axis=1) > 0
            samples_in_patient = sum(mask)

            temp = Y[mask, :, :]
            temp = K.eval(K.one_hot(temp, num_classes=2))
            data.append(X[mask, :, :])
            labels.append(temp)
            lens.append(samples_in_patient)
            inds.append(i)
        except:
            next

    X_fin = np.concatenate(data, axis=0)
    Y_fin = np.concatenate(labels, axis=0)
    X_fin = np.expand_dims(X_fin, axis=3)
    return X_fin, Y_fin

X_train, Y_train = load_doctor_data(indices=train_ind)
X_val, Y_val = load_doctor_data(indices=val_ind)

print(X_train.shape)
print(X_val.shape)

seed = 1
train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1)

input_generator = train_datagen.flow(
        X_train,
        batch_size=batch_size,
        seed=seed
)

label_generator = train_datagen.flow(
        Y_train,
        batch_size=batch_size,
        seed=seed
)

# Where to save logs and weights
weights_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-T1ce-lr-' + str(lr)\
               + '-n-' + str(X_train.shape[0]) + "-asgeirs_data_176.hdf5"
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-T1ce" \
          + '-lr-' + str(lr) + '-n-' + str(X_train.shape[0]) + '-asgeirs_data_176'

cp = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', mode='auto', verbose=1, period=1)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
tbc = TensorBoard(log_dir=log_dir)

tot_generator = zip(input_generator, label_generator)

unet.fit_generator(
        tot_generator,
        steps_per_epoch=num_batches_in_epoch,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=[cp, es, tbc],
        verbose=1
)

X_test, Y_test = load_doctor_data(test_ind)
yhat = unet.predict(X_test)
np.savez_compressed("predictions_asgeirs_test_data.npz", yhat)