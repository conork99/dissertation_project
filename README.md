# Hybrid Transformer Network for Diabetic Retinopathy Severity Grading and Lesion Segmentation

## A joint classification and segmentation network for automated diabetic retinopathy (DR) diagnosis. The network uses classification to provide a DR severity prediction, and segmentation to locate and identify four different types of retinal lesions associated with DR.

This project served as my dissertation research for my MSc Bioinformatics at the University of Liverpool. The network we created aims to automate DR severity grading while outputing lesion information, making it more clinically appropriate. Our model, which was inspired by the H2Former model developed by He et al. (2023), utilses a hybrid Transformer encoder which feeds both the classification and segmentation branches of the model. The segmentation branch, inspired by the PraNet model developed by Fan et al. (2020), implements Parallel Partial Decoders and Reverse Attention Modules. In order to assess the performance of our model we used adaptations of U-Net, Attention U-Net, U-Net++, and Dense U-Net models as baseline models, the codes for which are included in this repository. The complete final report on this research is also included in this repository. 

## Structure of our hybrid Transformer joint classification and segmentation model
![Model Image](https://github.com/conork99/dissertation_project/assets/135136497/b4483000-e9c9-4b43-ba58-8c4d67817f33)

## Navigating this respository
Under the folder "code_mres_conor" there is another folder with the same name which contains all the training and testing scripts for our model and all of the baselines. The codes for the model architectures are in the "lib" folder, found in the same place as the training and testing scripts. The pre-processed images that we used are included in the "h5py_FGADR" folder. Note: these images were obtained from the FGADR dataset created by Zhou et al. (2021).

## How to adapt this project for your own use
Ensure that the file and folder locations specified in the model architecture script files, as well as the training and testing script files, are modified to be specific to your device. Also, be sure to create an output folder in the appropriate location for the model to save to when running the training script. The folder will be referred to in the training scripts as model_grading_seg and was located within the first code_mres_conor folder. 

I hope this is useful.
