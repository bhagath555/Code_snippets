Useful directories
* testing
* testing_morph
* training
* training_morph
* ground_truth 

Useful files
* **image_morph_ops.m** : It read all the files from *testing* and *training* directory and apllies morphotgical operations and saves the images into *testing_morph* and *training_morph* directories.
* **main.m** : It reads the features from morphed images and extracts features and SVM trained with the data. Tested the images.
* **shading.m** : It removes the shading effect from the images.

* **crack_length.m** : To calculate the crack lenth.
* **IntersectionOverUnion.m** : To calculate the intersection over Union metric for classifier performance.
