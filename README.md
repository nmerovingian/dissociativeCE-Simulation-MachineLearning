# dissociativeCE-Simulation-MachineLearning
 Simulation and machine learning of dissociative CE reaction

The author use Python 3.7.3 with Anaconda. The C++ program require C++ 11 for std::thread. 

# Machine Learning 

Contains machine learning scripts using keras(deprecated) or tensorflow.keras. Machine learning packages like Numpy, Scikit-Learn and xgboost are required. Pandas, Matplotlib, Seaborn are required for data processing and visualization. Graphviz and ann_visualizer is used for visualization of neural networks.
All three methods mentioned in paper are included in this folder, including:



## Python Scripts for machine learning

* **Predict Constants.py** Neural network for method A mentioned in paper. It can predict equilibrium constant and rate constant from voltammogram.
* **Predict Voltammogram.py** Neural network for method B mentioned in paper. It can predict voltammogram given equilibrium constant, rate costant and scan rate. 
* **Predict Flux and Half Peak Potential.py** Neural network for Method C mentioned in paper. It can predict half peak potential and peak flux from equilibrium constants and rate constants.
* **Benchmarking Models.py** Benchmarking neural network in Method A with naive linear regression, random forest regresor and xgboost regressor.

**features.csv** The csv file containing features and targets of for Method A and Method B extraced from thousands of voltammograms in the *Training Data* folder.

## Folders

* *weights* This folder contains weights of neural networks from the authors. The machine learning program will use existing weights in this folder if available. If not, will start training and weights will be saved in this folder
* *Training Data* The folder training data is stored.
* *Test Data* The folder testing data is stored. 

## Python Scripts for visualization of machine learning results








# Simulation C++

Simulation program of dissociative CE reaction using C++. Good for users loving C++.

# Simulation Python

Simulation program of dissociative CE reaction using Python. Good for users loving Python. Ordinary packages like Numpy, Scipy, Matplotlib are required. 


# Visualizing Voltammograms
A bunch of useful scripts for visualization of voltammogram files, plotting concentration profiels, plotting Tafel plots and analysis.  Sample data is included for reference. helper.py is a helper scripts for data analysis and visualization. Includes scripts to locate voltammogram files, to parse files and formatter for matplotlib.



**requirement.txt**  The python package the author used on Python 3.7.3 conda environment.


___

You may cite the paper as acknowledgement to the authors if you found this helpful. 

Please cite: 


For general enquiry, please email [Professor Richard Compton]<mailto:richard.compton@chem.ox.ac.uk>