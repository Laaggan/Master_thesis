base_path = ''

# Much cleaner loading of patients
train_data = load_patients(i=0, j=2, num_classes=4, base_path=base_path)
val_data = load_patients(i=3, j=4, num_classes=4, base_path=base_path)

X_train = train_data[0]
Y_train = train_data[1]

X_val = val_data[0]
Y_val = val_data[1]

# The path to where to save weights and initialize ModelCheckpoint
weights_path = config['weights_path']
from keras.callbacks import ModelCheckpoint, EarlyStopping
MyModelCheckPoint = ModelCheckpoint(weights_path, verbose=0, save_weights_only=True, period=1)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

if config['keep_training'] == True:
    # Keep training on the old weights
    my_unet = unet_res(input_size = (240, 240, 4), num_classes = 4)
    my_unet.load_weights(weights_path)
else:
    # Initialize network
    my_unet = unet_res(input_size = (240, 240, 4), num_classes = 4)
    config['keep_training'] = True

assert not np.any(np.isnan(X_train)), 'Input contain nans'

Y_train = Y_train.reshape(Y_train.shape[0], -1, 4)
validation_data = (X_val, Y_val.reshape(Y_val.shape[0], -1, 4))

# Returns an object with accuracy and loss
history = my_unet.fit(x=X_train, 
                      y=Y_train, 
                      batch_size=64,
                      epochs=100, 
                      verbose=1, 
                      callbacks=[CallbackJSON(config=config), MyModelCheckPoint, es],
                      validation_split=0.0, 
                      validation_data=validation_data, 
                      shuffle=True, 
                      class_weight=None, 
                      sample_weight=None, 
                      initial_epoch=0, 
                      steps_per_epoch=None, 
                      validation_steps=None, 
                      validation_freq=1)
