# Fourier neural operator for real-time simulation of 3D dynamic urban microclimate (FNO-3DUM)

This is the repository for the paper ([**Download**](https://authors.elsevier.com/a/1iAqO1HudNBsbT), [arXiv](https://doi.org/10.48550/arXiv.2308.03985)):

_W. Peng, S. Qin, S. Yang, J. Wang, X. Liu, and L. Wang*. 2023. Fourier neural operator for real-time simulation of 3D dynamic urban microclimate. Building and Environment. Volume 248, 111063._
https://doi.org/10.1016/j.buildenv.2023.111063

Global urbanization has underscored the significance of urban microclimates for human comfort, health, and building/urban energy efficiency. However, analyzing urban microclimates requires considering a complex array of outdoor parameters within computational domains at the city scale over a longer period than indoors. As a result, numerical methods like Computational Fluid Dynamics (CFD) become computationally expensive when evaluating the impact of urban microclimates. 

In this work, we apply the FNO network for real-time 3D urban microclimate simulation. The FNO model has a 0.3% one-step prediction error and a maximum error of 5% when applied to unseen data with different wind directions. A real-time simulation of urban microclimates in 3D is possible with the FNO approach, which is 25 times faster than the traditional numerical solver.

**Visualization**

https://github.com/CUBELeonwang/FNO-3DUM/assets/67432536/73a7923a-1df2-4307-a473-cb0ea46084b9

## Data description

[**Download full dataset**](https://www.kaggle.com/datasets/shaoxiangqin/cityffd-3d-urban-wind-simulation-niigata)

The dataset of 3D urban wind simulation data of Niigata is generated from CityFFD. A total of 1200 steps of wind simulation were executed. The dataset contains four wind directions of data. Data for the _west_ and _north_ winds include _all 1200_ simulation steps. Data for the _east_ and _south_ winds include the _last 50_ steps of the simulation. Each step of the data is a _200 * 200 * 150_ array with 32-bit precision and is stored as a numpy file.

## Model training and testing

Requirements: Pytorch 1.8.0

To train the FNO model, execute the `train_fno.py`. Within the configs section of the code, you can specify model hyperparameters, define paths for loading the dataset and saving the model, and set the dataset loading method. The configuration allows model training on either a single wind direction or multiple wind directions. 

`test_fno.py` can be used to evaluate the performance of a trained model on data from four wind directions.

The implementation is based on the work of Li et al.:
https://github.com/neuraloperator/neuraloperator/tree/master
