# cubeml
CubeML is a Python package to support semantic segmentation of reflectance-hyperspectral images (hypercubes) using diverse machine learning (ML) approaches. An Automated Machine Learning (AutoML) approach is provided to automate hyperparameter tuning for some of these approaches.

In this readme, we will walk through basic use of CubeML for use in semantic segmentation to support research. 

This is an early work-in-progress as of August 2023. A more complete beta with support for more model types, checks, and more documentation is forthcoming.

## 1. Build a training set of labels
Before we use CubeML, we need to label our images that will go into the training set. We provide a script to generate these false color images for batches of hyperspectral images.

These false color images can then be loaded into [LabelStudio](https://labelstud.io) and annotated using the "brush tool" as demonstrated on this [tutorial slide](https://github.com/naglemi/cubeml/blob/main/notebooks/LabelStudio_tutorial_slide.png).

## 2. Load hyperspectral data and labels as `TrainingData` class
Let's first initialize the TrainingData object, using the json file and png files output from LabelStudio, along with our raw hyperspectral data. It is suggested to set `normalize_features` to `True` for best performance when brightness may fluctuate.

``` python
training_data = TrainingData(
    json_file=json_file,
    img_directory=hypercube_dir,
    png_directory=png_directory,
    normalize_features=True)
```

Do you have a very large and imbalanced training dataset (e.g., a million pixels of class X, and only 50,000 for class Y)? If so, taking a stratified sample will balance your dataset. Each class will be represented by the same number of pixels as the class with the least representation.

``` python
training_data.stratified_sample()
```

The `TrainingData` object can be easily saved out for later use.

``` python
pickle.dump(training_data, open(filename, 'wb'))
```

We can plot the mean spectra for each class in via `TrainingData.plot_spectra()`. You can see an example plot [here](https://github.com/naglemi/cubeml/blob/main/plot_examples/plot_spectra_output.png?raw=true)

## 3. Training models

### 3A. Training one model at a time as a `CubeLearner`

#### 3A.i. Basic use

Here is a basic example, in which we'll use default hyperparameters to train Linear Discriminant Analysis (LDA) model.
``` python
my_model = CubeLearner(training_data,
                       model_type = "LDA")
my_model.fit()
```

#### 3.A.ii. AutoML for fine-tuning models

Several of the ML methods used here, including Gradient Boosting Classifiers (GBC), Decision Tree Classifiers (DTC) and Random Forests (RF) can undergo an automated hyperparameter optimization process. The suggested approach is to first employ a grid search, then fine-tune pseudo-optimized parameters output from the grid search using a genetic optimization algorithm.

``` python
my_model = CubeLearner(training_data, model_type="RF")

# First run Grid Search to pseudo-optimize parameters
my_model.fit(automl="grid")

# Based on pseudo-optimized parameters, build parameter ranges for fine-tuning
param_ranges = get_param_ranges(learner_dict, model_type)

my_model.fit(automl="genetic", param_ranges=param_ranges)
```

### 3B. Training and comparing model performance with `CubeSchool`

We can train many models with the same `TrainingData` and compare the performance of various algorithms.

``` python
model_types = ["PCA", "LDA", "RF", "GBC", "ABC", "LR", "GNB", "DTC", "TNN"]

school = CubeSchool(training_data, model_types, colors)
school.run()
```

Using `CubeSchool.presentable_table` and `CubeSchool.compare_inferences`, we can compare then accuracies of various models by their Intersection of Union (IoU) statistics and by looking at inference labels for specific images side-by-side.

## Acknowledgements
We thank the National Science Foundation Plant Genome Research Program for support (IOS #1546900, Analysis of genes affecting plant regeneration and transformation in poplar), and members of GREAT TREES Research Cooperative at OSU for its support of the Strauss laboratory.

This project was inspired in large part by this prior work. Several important parts of code related to data parsing and spectra plotting were derived from this work.

Miao, C., Xu, Z., Rodene, E., Yang, J., & Schnable, J. C. (2020). Semantic segmentation of sorghum using hyperspectral data identifies genetic associations. Plant Phenomics, 2020.




