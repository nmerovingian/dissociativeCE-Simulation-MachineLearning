# dissociativeCE-Simulation-MachineLearning
 Simulation and machine learning of dissociative CE reaction. Simulation assumes 1-D simulation of reversible electron transfer on a spherical electorde coupled with dissociative preceding chemical reaction. A set of machine learning prorgams are offered to analyze the relationship between the rate constants and voltammograms. Good for analytical purpose.  

The author used Python 3.7.3 in Anaconda environment. The C++ program requires C++ 11 for std::thread. 

# Machine Learning 

Contains machine learning scripts using keras(deprecated) or tensorflow.keras. Machine learning packages like Numpy, Scikit-Learn and xgboost are required. Pandas, Matplotlib, Seaborn are required for data processing and visualization. Graphviz and ann_visualizer is used for visualization of neural networks.
All three methods mentioned in paper are included in this folder, including:



### Python Scripts for machine learning

* **Predict Constants.py** Neural network for method A mentioned in paper. It can predict equilibrium constant and rate constant from voltammogram.
* **Predict Voltammogram.py** Neural network for method B mentioned in paper. It can predict voltammogram given equilibrium constant, rate costant and scan rate. 
* **Predict Flux and Half Peak Potential.py** Neural network for Method C mentioned in paper. It can predict half peak potential and peak flux from equilibrium constants and rate constants.
* **Benchmarking Models.py** Benchmarking neural network in Method A with naive linear regression, random forest regresor and xgboost regressor.

### Python Scripts for visualization of machine learning results
* **Benchmarking Models.py** Benchmarking model B with naive linear regression, XGB regressor and random forest.
* **ContourPlotData predict constants-Dimensionless.py** Visualization of predicting constants 
* **ContourPlotData predict half peak potential and flux - Dimensionless.py**  Visualization of predicting half peak potential and peak flux
* **Histogram predict voltammogram Dimensionless.py**  Visualization of predicting voltammograms

### Folders

* *weights* This folder contains weights of neural networks from the authors. The machine learning program will use existing weights in this folder if available. If not, will start training and weights will be saved in this folder
* *Training Data* The folder training data is stored. This folder is intentionally left empty due to the large number (more than 10,000) of training voltammogram files used. The user can generate their own training data using the simulation programs. 
* *Test Data* The folder testing data is stored.  This folder is intentionally left empty. The user should generate their own testing voltammograms.


**features.csv** The csv file containing features and targets of for machine learning using Method A and Method B, which is extraced from thousands of voltammograms in the *Training Data* folder.




# Simulation, C++

Simulation program of dissociative CE reaction using C++. Good for users loving C++. Simulation of other electrochemical reactions including CE,EC,EC2 can be found [here](https://github.com/nmerovingian/CE_Dissociative-CE_EC_EC2_Reaction_Simulation).

# Simulation, Python

Simulation program of dissociative CE reaction using Python. Good for users loving Python. Ordinary packages like Numpy, Scipy, and Matplotlib are required. 


# Visualizing Voltammograms
A bunch of useful scripts for visualization of voltammogram files, plotting concentration profiels, plotting Tafel plots and analysis.  Sample data is included for reference. *helper.py* is a helper scripts for data analysis and visualization, which includes scripts to locate voltammogram files, to parse files and formatter for matplotlib.



**requirement.txt**  The python packages the author used in a Python 3.7.3 conda environment.


___

You may cite the paper as acknowledgement to the authors if you found this helpful. 

Please cite: 


For general enquiry, please email [Professor Richard Compton](mailto:richard.compton@chem.ox.ac.uk)