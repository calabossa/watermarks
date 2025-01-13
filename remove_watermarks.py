# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:32:31 2024

@author: boazs
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, UpSampling2D, Dropout, 
       BatchNormalization, Conv2DTranspose,  Activation, Concatenate, Input)
from tensorflow.keras.models import Sequential
from tensorflow import cast
from tensorflow import float16, float32
from tensorflow.image import ssim
from tensorflow import reduce_mean
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.saved_model import SaveOptions
from keras.saving import load_model


## Autoencoder and U-net for watermark removal

select_net_type = 'Unet' # 'Unet' or 'autoencoder'

model_save_path = r"path\to\model\unet_best_checkpoint.keras"
results_save_path = r"path\to\results\history_Dec05.json"
path_to_temp_memmap_files = r"path\to\temp\files" # to handle large dataset on my laptop, stims will be saved as memmap files here

## Set paths to Train, Valid and Test dataset locations
train_no_water_path = r"path\to\pics\TRAIN\NO_watermark"
train_water_path = r"Cpath\to\pics\TRAIN\watermarked"
valid_no_water_path = r"path\to\pics\VALID\NO_watermark"
valid_water_path = r"Cpath\to\pics\VALID\watermarked"
test_no_water_path = r"path\to\pics\TEST\NO_watermark"
test_water_path = r"path\to\pics\TEST\watermarked"

def nearest_divisible(numb, denominator):
    X = np.arange(denominator, numb+denominator, denominator)
    vct = abs(numb - X)
    idx = np.where(vct == min(vct))
    idx = idx[0][0] + 1
    return idx * denominator

# what size should the images be resized to? They all have the same height: 350pxl
fileList = glob.glob(train_no_water_path +'\\' + '*')
dimension_count = []
for i in fileList:
    img = cv2.imread(i)
    dimension_count.append(img.shape[1])
# find the most common value for width:
mode = max(set(dimension_count), key=dimension_count.count)  # it should be 525
new_size = (mode, mode) # or, 450 can be a compromize between narrowing too much and widening too much (following visual inspection)
rat_reduction = 2 # by how much to reduce
new_size = [round(new_size[0]/rat_reduction), round(new_size[1]/rat_reduction)]
new_size[0] = nearest_divisible(new_size[0], 32)
new_size[1] = nearest_divisible(new_size[1], 32)
new_size = tuple(new_size)

## arranging the stimuli into matrices, after size and quality reduction (have no choiuce, it's too big for my local laptop memory)

shape = (len(fileList), new_size[1], new_size[0], 3)
dtype = 'float32'

# Build the NO-watermark TRAIN data matrix
filename = path_to_temp_memmap_files + r'\images_memmap_NON_water.dat'
memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

for i, path in enumerate(fileList):
    img = cv2.imread(path)
    img = cv2.resize(img, new_size)
    # img = np.round(img/255,2)
    img = cast(img, dtype=float32)
    img_array = np.array(img, dtype='float32') / 255.0  # Convert to float16 and normalise
    memmap_array[i] = img_array
print(memmap_array.shape)

# Build the watermark TRAIN data matrix
filename_water = path_to_temp_memmap_files + r'\images_memmap_water.dat'
memmap_array_water = np.memmap(filename_water, dtype=dtype, mode='w+', shape=shape)

for i, path in enumerate(fileList):
    figname = os.path.basename(path)
    expected_file = train_water_path + '\\' + figname
    img = cv2.imread(expected_file)
    img = cv2.resize(img, new_size)
    # img = np.round(img/255,2)
    img = cast(img, dtype=float16)
    img_array = np.array(img, dtype='float16') / 255.0  # Convert to float16 and normalise
    memmap_array_water[i] = img_array
print(memmap_array_water.shape)


# Validation database:

fileListValid = glob.glob(valid_no_water_path +'\\' + '*')
shapeV =  (len(fileListValid), new_size[1], new_size[0], 3)

filename_V = path_to_temp_memmap_files + r'\images_memmap_VALID_NON_water.dat'
memmap_array_valid = np.memmap(filename_V, dtype=dtype, mode='w+', shape=shapeV)
for i, path in enumerate(fileListValid):
    img = cv2.imread(path)
    img = cv2.resize(img, new_size)
    img = cast(img, dtype=float16)
    img_array = np.array(img, dtype='float16') / 255.0  # Convert to float16 and normalise
    memmap_array_valid[i] = img_array
print(memmap_array_valid.shape)

# Build the watermark VALID data matrix
filename_water_V = path_to_temp_memmap_files + r'\images_memmap_VALID_water.dat'
memmap_array_water_valid = np.memmap(filename_water_V, dtype=dtype, mode='w+', shape=shapeV)

for i, path in enumerate(fileListValid):
    figname = os.path.basename(path)
    expected_file = valid_water_path + '\\' + figname
    img = cv2.imread(expected_file)
    img = cv2.resize(img, new_size)
    # img = np.round(img/255,2)
    img = cast(img, dtype=float16)
    img_array = np.array(img, dtype='float16') / 255.0  # Convert to float16 and normalise
    memmap_array_water_valid[i] = img_array
print(memmap_array_water_valid.shape)


## External test dataset

fileListTest = glob.glob(test_no_water_path +'\\' + '*')
shapeTest =  (len(fileListTest), new_size[1], new_size[0], 3)

# Build the watermark TEST data matrix
filename_test = path_to_temp_memmap_files + r'\images_memmap_TEST_NON_water.dat'
memmap_array_test = np.memmap(filename_test, dtype=dtype, mode='w+', shape=shapeTest)

for i, path in enumerate(fileListTest):
    img = cv2.imread(path)
    img = cv2.resize(img, new_size)
    # img = np.round(img/255,2)
    img = cast(img, dtype=float16)
    img_array = np.array(img, dtype='float16') / 255.0  # Convert to float16 and normalise
    memmap_array_test[i] = img_array
print(memmap_array_test.shape)

# Build the watermark TEST data matrix
filename_water_test = path_to_temp_memmap_files + r'\images_memmap_TEST_water.dat'
memmap_array_water_test = np.memmap(filename_water_test, dtype=dtype, mode='w+', shape=shapeTest)

for i, path in enumerate(fileListTest):
    figname = os.path.basename(path)
    expected_file = test_water_path + '\\' + figname
    img = cv2.imread(expected_file)
    img = cv2.resize(img, new_size)
    # img = np.round(img/255,2)
    img = cast(img, dtype=float16)
    img_array = np.array(img, dtype='float16') / 255.0  # Convert to float16 and normalise
    memmap_array_water_test[i] = img_array
print(memmap_array_water_test.shape)


### === MODELS ===


## verify correct model selection
runthenet = True
cnfrm = input(f'\n\n === The model type selected is:  {select_net_type}. Press y and enter to confirm:  ')
if cnfrm != 'y':
    print('ABORTING RUN')
    runthenet = False


if select_net_type.lower() == 'autoencoder' and runthenet:

    ## Building an AUTOENCODER
    
    def ssim_loss(y_true, y_pred):
        return 1 - reduce_mean(ssim(y_true, y_pred, max_val=1.0))
    
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_in_shape=(new_size[0], new_size[1], 3))) # img.shape
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # going back up
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    # Output layer with sigmoid (to avoid flattening out pixal and maybe solve the problem of color loss...)
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same')) # relu
    
    ## Compile 
    
    model.compile(optimizer='adam', loss=ssim_loss, metrics=['accuracy'])  # 'mean_squared_error' 'mean_squared_error'   ,
    model.summary()
        
    ## Select some params
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                                  factor=0.4, # factor to reduce LR by  # try also: 0.5
                                  patience=4,  # 5  
                                  min_lr=1e-5, # Lower bound on learning rate
                                  verbose=1)  
         
    callbacks = [reduce_lr, early_stopping, checkpoint]  
        
    
    ## FIT:
        
    history = model.fit(memmap_array_water, memmap_array, 
                        epochs=50,
                        batch_size=200, 
                        shuffle=True, 
                        validation_data=(memmap_array_water_valid, memmap_array_valid),
                        callbacks=callbacks)
    
    with open(results_save_path, 'w') as f:
        json.dump(history.history, f)


elif (select_net_type.lower() == 'unet' or select_net_type.lower() == 'u-net') and runthenet:
    
    
    ## Define U-NET blocks as functions
    
    def conv_block(data_in, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(data_in)
        x = BatchNormalization()(x) 
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)  
        x = Activation("relu")(x)
    
        return x

    # Encoder block: Conv block followed by maxpooling
    def encoder_block(data_in, num_filters):
        x = conv_block(data_in, num_filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p   
    
    # Decoder block
    # "skip" features get inputs (data_in) from the encoder block outputs ("skip" connections)
    
    def decoder_block(data_in, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(data_in)
        x = Concatenate()([x, skip_features])
        x = conv_block(x, num_filters)
        return x
    
    
    
    ## building U-NET: 
    
    def build_unet(input_shape):
        inputs = Input(input_shape) # Input is a layer imported from keras
    
        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        s3, p3 = encoder_block(p2, 256)
        s4, p4 = encoder_block(p3, 512)
    
        b1 = conv_block(p4, 1024) # the bridge / bottleneck 
    
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)
    
        outputs = Conv2D(3, 1, padding="same", activation="sigmoid")(d4)  # Binary (can be multiclass)
    
        model = Model(inputs, outputs, name="U-Net")
        
        return model
    
    # run the model using the custom functions
    
    input_shape = memmap_array_water.shape[1:]
    
    model = build_unet(input_shape)
    model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    batch_size = 200
    steps_per_epoch = 100 # 3*(len(memmap_array))//batch_size

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        model_save_path, 
        save_best_only=True, 
        monitor='val_loss', 
        verbose=1,
        save_freq='epoch',
        )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                                  factor=0.4, 
                                  patience=3,  
                                  min_lr=1e-5, 
                                  verbose=1)  
    callbacks = [reduce_lr, early_stopping, checkpoint] #

    # FIT
    
    history = model.fit(
    x = memmap_array_water, y=memmap_array,
    batch_size=batch_size, 
    epochs=4,
    verbose=1,  
    validation_data=(memmap_array_water_valid, memmap_array_valid),
    steps_per_epoch=steps_per_epoch, 
    callbacks=callbacks
)
    
    model.save(model_save_path)

## Load model if needed

#   see below

##

## Predict

if runthenet:

    pred = model.predict(memmap_array_water_test)
    
    
    # show some examples
    nmb = np.random.randint(0, len(pred))
    img1 = cv2.cvtColor(pred[nmb], cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.title('predicted')
    img2 = cv2.cvtColor(memmap_array_water_test[nmb], cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img2)
    plt.title('Original')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(img2)
    plt.title('Original')
    plt.subplot(2,1,2)
    plt.imshow(img1)
    plt.title('predicted')
    


## !! To load the model with the custom loss function inside:
    
if False:
        
    loaded_model = load_model(r"path\to\best_model_so_far.keras", custom_objects={'ssim_loss': ssim_loss})
    # OR:
    loaded_model = load_model(r"path\to\unet_best_checkpoint_Dec19.keras")


    # If needed re-evaluate the model
    loss, accuracy = loaded_model.evaluate(memmap_array_valid, memmap_array_water_valid)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    
    # predict with the loaded model -
    pred = loaded_model.predict(memmap_array_water_test)

