# RNN-Project
Project
James Hooper ~ NETID: jah171230
Hritik Panchasara ~ NETID: hhp160130
------------------------------------------------------------------------------------------------------------------------------------
- For this assignment we used PyCharm to create/edit/run the code. 
- In PyCharm the code should run by simply pressing the Run dropdown, then clicking run making sure you are running LSTM_RNN_FINAL.py
- The dataset is held in one of our utdallas box accounts with a direct link used in the code.
------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS & PARAMETERS
- In if __name__ == '__main__' , you can select the following parameters: all attributes, print initial data graphs, and the train/test split value.
- In process_data(all_atr, print_data_graphs), you can alter the resampling parameters in res_df = df.resmaple('h').mean() for example by changing 'h' to 'D' for day or changing mean() to sum().
- In LSTM_Model(rf_data, scaler, split), you can alter the model parameters to add more layers (make sure last LSTM layer has return_sequences = False), change the dropout rate, and edit the units of the LSTM layers. In the lstm_model.fit(...) you can change the number of epochs.
- Some issues occur with too big of sample size or k-value due to memory allocation. Not sure how to fix this it may be a problem dependent on system or IDE.
------------------------------------------------------------------------------------------------------------------------------------
MAJOR FUNCTIONS PURPOSES:
~ series_to_supervised: transforms data from time-series to supervised to be run through model
~ process_data: grab the data, pre-process the attributes, resample the data over a certain period of time steps, scale the data, and reframe the data.
~ LSTM_Model: split the reframed data into train/test, intialize LSTM layers for model, graph the loss function of the train dataset and the validation test dataset, print the RMSE & R^2 values, and graph the actual vs forecasted data for both the train & test datasets ran through the model.
~ __main__: a driver to execute all the functions with some overarching parameter choices
------------------------------------------------------------------------------------------------------------------------------------
Link to Datasets/Images:
~ https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
- Reminder that these are all being housed in one of our utdallas box storages for ease of use within PyCharm.
------------------------------------------------------------------------------------------------------------------------------------
Libraries Used:
'''
Packages:
numpy, pandas, scikit-learn, seaborn, matplotlib, tensorflow, keras
'''
from tensorflow import keras as krs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
------------------------------------------------------------------------------------------------------------------------------------
Just in case. To import libraries/packages in PyCharm.
- Go to File.
- Press Settings.
- Press Project drop down.
- Press Project Interpreter.
- Press the plus sign on the top right box, should be to the right of where it says "Latest Version".
- Search and Install packages as needed.
- For this assignment the packages are: numpy, pandas, scikit-learn, seaborn, matplotlib, tensorflow,  and keras.
~ KEY REMINDER: We used Anaconda Navigator to implement a viable environment to run tensorflow. This can change with versions of numpy where one might have to downgrade the version to make everything work. This project was done in Python 3.7 due to the limitations of Tensorflow compatibility. There may be warnings based on your own system (CPU, GPU, etc.) and settings.
