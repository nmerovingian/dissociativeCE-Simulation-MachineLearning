# dissociativeCE-Simulation-MachineLearning
 Simulation and machine learning of dissociative CE reaction


# Machine Learning 

Contains machine learning scripts using keras(deprecated) or tensorflow.keras. Machine learning packages like Numpy, Scikit-Learn and xgboost are required. Pandas, Matplotlib, Seaborn are required for data processing and visualization. Graphviz and ann_visualizer is used for visualization of neural networks.
All three methods mentioned in paper are included in this folder, including:

## Python Scripts for machine learning

* Predict Constants.py Neural network for method A mentioned in paper. It can predict equilibrium constant and rate constant from voltammogram.
* Predict Voltammogram.py Neural network for method B mentioned in paper. It can predict voltammogram given equilibrium constant, rate costant and scan rate. 
* Predict Flux and Half Peak Potential.py Neural network for Method C mentioned in paper. It can predict half peak potential and peak flux from equilibrium constants and rate constants.

## Python Scripts for data analysis and visualization


## Folders

* weights This folder contains weights of neural networks from the authors. The machine learning program will use existing weights in this folder if available. If not, will start training and weights will be saved in this folder





# Simulation C++

Simulation program of dissociative CE reaction using C++. Good for users loving C++.

# Simulation Python

Simulation program of dissociative CE reaction using Python. Good for users loving Python. Ordinary packages like Numpy, Scipy, Matplotlib are required. 
