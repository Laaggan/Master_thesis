# Scripts for training U-Nets for BraTS challenge

The scripts that were used to train the 
networks in the master thesis are the following

1. 20191204-unet_with_data_augmentation.py
2. 20191209-modalities_separately_data_augmentation.py
3. 20191217-new_sensor_fusion_v2.py
4. 20200108-unet_sensor_fused_pretrained.py

Where script 1 is referred to as the non-sensor fused in the thesis. \
To train the sensor fused network which is trained end-to-end one runs script 3. \
To train the pretrained sensor fused network first one runs script 2 to train each 
modality on the task of brain tumor and later one trains a sensor fusion block by 
running script 4.

# Data setup
The data setup is that one should have the MICCAI BraTS-dataset
converted into numpy in the repo for the scripts to work.

# Results
Here follows a few examples of the results of the different networks
that were implemented

![Patient 13](results/patient-13.png)

![Patient 106](results/patient-106.png)

![Patient 121](results/patient-121.png)