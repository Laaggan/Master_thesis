import nibabel as nib
import numpy as np
import os

def save_numpy_as_nifti(segmentation, brain):
    '''segmentation: should be a numpy array with structure [number of patients, pixelsX, pixelsY, slices]
        and coverts it to to number of patients nifti files

        brain: should be a numpy array of shape [slices, pixelsX, pixelsY, modalities]
    '''
    path_to_nifti_file = ''
    fixed_data = nib.load(path_to_nifti_file)

    #nr_patients = segmentation.shape[0]/155
    r = segmentation.shape[0]
    new_header = fixed_data.header.copy()
    new_extra = fixed_data.extra.copy()
    new_affine = fixed_data.affine.copy()


    for i in range(r):
        segment = nib.Nifti1Image(segmentation[i,:,:,:], affine=new_affine, header=new_header, extra=new_extra)
        nib.save(segment,'pred_seg'+ str(i) +'.nii.gz')

        nifti_brain_t1 = nib.Nifti1Image(brain[i*155:i*155+154, :, :, 0], affine=new_affine, header=new_header, extra=new_extra)
        nib.save( nifti_brain_t1, 't1' + str(i) + '.nii.gz')
        nifti_brain_t1ce = nib.Nifti1Image(brain[i*155:i*155+154, :, :, 1], affine=new_affine, header=new_header, extra=new_extra)
        nib.save(nifti_brain_t1ce, 't1ce' + str(i) + '.nii.gz')
        nifti_brain_t2 = nib.Nifti1Image(brain[i*155:i*155+154, :, :, 2], affine=new_affine, header=new_header, extra=new_extra)
        nib.save(nifti_brain_t2, 't2' + str(i) + '.nii.gz')
        nifti_brain_flair = nib.Nifti1Image(brain[i*155:i*155+154, :, :, 3], affine=new_affine, header=new_header, extra=new_extra)
        nib.save(nifti_brain_flair, 'flair' + str(i) + '.nii.gz')


