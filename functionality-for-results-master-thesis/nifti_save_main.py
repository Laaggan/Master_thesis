from save_nifti import *
from my_lib import *

img2 = nib.load('/Users/carlrosengren/Documents/EXJOBB/test_18_11/Master_thesis/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_18_1/BraTS19_2013_18_1_seg.nii.gz')
img2_data = img2.get_fdata()
img3 = nib.load('/Users/carlrosengren/Documents/EXJOBB/test_18_11/Master_thesis/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_22_1/BraTS19_2013_22_1_seg.nii.gz')
img3_data = img3.get_fdata()
img4 = nib.load('/Users/carlrosengren/Documents/EXJOBB/test_18_11/Master_thesis/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_AOH_1/BraTS19_CBICA_AOH_1_seg.nii.gz')
img4_data = img4.get_fdata()
img5 = nib.load('/Users/carlrosengren/Documents/EXJOBB/test_18_11/Master_thesis/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA09_402_1/BraTS19_TCIA09_402_1_seg.nii.gz')
img5_data = img5.get_fdata()

patients = np.full((2,176,176,155),0)

n = 335
np.random.seed(42)
ind = np.arange(n)
np.random.shuffle(ind)
ind1 = int(np.ceil(len(ind) * 0.7))
ind2 = int(np.ceil(len(ind) * 0.85))

train_ind = ind[0:ind1]
val_ind = ind[ind1:237]

path_to_folder = '/Users/carlrosengren/Downloads/numpy_patients'
X_val_raw, Y_val = load_patients_numpy(path_to_folder, val_ind, cropping=True)

patients[0,:,:,:] = img2_data[40:216,40:216,:]
patients[1,:,:,:] = img3_data[40:216,40:216,:]
#patients[2,:,:,:] = img4_data
#patients[3,:,:,:] = img5_data

save_numpy_as_nifti(patients, X_val_raw)