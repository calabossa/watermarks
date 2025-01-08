This project is designed to practive autoencoders and U-nets uisng Keras-Tensorflow, as well as the openCV library. It includes a dedicated network for the removal of watermarks from color images. 
The network can be selected from a choice of: 
1. Autoencoder (faster)
2. U-net (slower, more parameters)
   
The input images should consist of a set of images and an equivalent set of the same images, with the same names, watermarked and saved in a separate folder.
The code uses 2 such pairs of folders: Training and Validation. An additional Test is used for quantification and visual inspection. I used a ratio of 70-30 for training-validation,
and a set of 1000 image pairs for Testing. 
If you do not have a database of images, there is also a code for the creation of random patterns of watermarks. This code takes any amount of images provided to it,
and adds a randomized watermark. The watermarks are randomized on several parameters such as characters (what will be written), font size, placement, color and degree of transparency.
When running the networks, the autoencoder functioned well only when increasing the dataset (training + validation) to above 20k images.

I tried inserting as many comments as possible in the codes, for clarity. 
A code for splitting training and validation is not provided here. 
