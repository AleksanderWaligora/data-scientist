# Comparison of classification algorithms in marketing forecasting
Data Science project comparing the efficiency of machine learning classification algorithms in forecasting results of bank marketing campaign. Marketing results are defined as information wheter client bought marketed product or no. This was done by going through various steps of Data Science Lifecycle ,which included training multiple machine learning models on different classification algorithms and comparing them. 

Data set used was obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

Project is split into multiple Python notebooks in order to segment each step of project.

Classification algorithms tested in this projects:
- Logistic regression
- K-nearest neighbors
- Naive Bayes classifier
- Decision tree classifier
- Random forest classifier
- Support vector machine classifier

## Project files

### Main files
 Main files used in project realisation, listed in order of their execution: 
- `data_evaluation.ipynb`: Data evaluation set features by their visualisation.
- `data_preparation.ipynb`: Data preparation for modeling by using techniques like removal of 
- `data_splitting.ipynb`: Splitting prepared data set into training, validation and testing data sets.
- `modeling.ipynb`: Training and saving multiple machine learning models.
- `model_evaluation.ipynb`: Evaluation of trained models by comparing their various metrics like accuracy, f1 scores and training time.

### Data files
 Directories containing original data set and data sets created during project realisation, saved in `.csv` format:
- `original/`: Original data set downloaded from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).
- `clean/`: Data after preparation using `data_preparation` file.
- `split/`: Data after splitting using `data_splitting` file.

### Helpers
Python files containing helper function used in Main files:
- `visualisation_helpers.py`: Functions used to help visualise data.
- `preparation_helpers.py`: Function used to help with data preparations.
- `modeling_helpers.py`: Functions used during training and evaluation of models.

### Results
Files containing project results:
- `model_metrics.png`: Plots showing comparison of models metrics.
- `train_time.csv`: File containing model training times.
