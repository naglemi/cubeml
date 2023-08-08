# cubeml
CubeML is a Python package to support semantic segmentation of reflectance-hyperspectral images (hypercubes) using diverse machine learning (ML) approaches. In Oregon State University's Forest Biotechnology Laboratory, we have long relied on semantic segmentation of RGB images. However, the reflectance-hyperspectral approach offers key advantages in accelerating training: 1) We don't need to label every single pixel in every training image; and 2) We can apply relatively more simple ML/DL methods that don't require very long period of training and hyperparameter optimization.

In this readme, we will walk through basic use of CubeML for use in semantic segmentation to support research.

## 1. Build a training set of labels
Before we use CubeML, we need to label our images that will go into the training set. We provide a script to generate these false color images for batches of hyperspectral images.

These false color images can then be loaded into [LabelStudio](https://labelstud.io) and annotated using the "brush tool" as demonstrated on this [tutorial slide](https://github.com/naglemi/cubeml/blob/main/notebooks/LabelStudio_tutorial_slide.png).

## 2. Load hyperspectral data and labels as `TrainingData` class
Let's first initialize the TrainingData object, using the json file and png files output from LabelStudio, along with our raw hyperspectral data. It is suggested to set `normalize_features` to `True` for best performance when brightness may fluctuate.

```
training_data = TrainingData(
    json_file=json_file,
    img_directory=hypercube_dir,
    png_directory=png_directory,
    normalize_features=True)
```

Do you have a very large and imbalanced training dataset (e.g., a million pixels of class X, and only 50,000 for class Y)? If so, taking a stratified sample will balance your dataset. Each class will be represented by the same number of pixels as the class with the least representation.

`training_data.stratified_sample()`

The `TrainingData` object can be easily saved out for later use.

`pickle.dump(training_data, open(filename, 'wb'))`

We can plot the mean spectra for each class in via `TrainingData.plot_spectra()`:

![plot_spectra_example](https://github.com/naglemi/cubeml/blob/main/plot_examples/plot_spectra_output.png?raw=true | width = 300)

## 3. Training models

### 3A. Training one model at a time as a `CubeLearner`

### 3B. Training and comparing model performance with `CubeSchool`

## Acknowledgements

This is an early work-in-progress as of July 2023. A complete beta along with documentation, more suitable for public use, is forthcoming.


