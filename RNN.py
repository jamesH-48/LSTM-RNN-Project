'''
Packages:
numpy, pandas, scikit-learn, seaborn, matplotlib, tensorflow, keras
'''
import tensorflow as tf
from tensorflow import keras as krs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error

'''
Function Source:
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    Alter a time series dataset to be a supervised learning dataset
    :param data: Sequential Dataset
    :param n_in: Number of lag observations as input
    :param n_out: Number of observations as output
    :param dropnan: Bool variable to determine if to drop NaN values
    :return: Supervised Dataset for Learning
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input Sequence ((t-n),...,(t-1))
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range (n_vars)]
    # Output/Forecast Sequence ((t),...,(t+n))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Concatenate Together
    final_df = pd.concat(cols, axis=1)
    final_df.columns = names
    # Drop rows that contain NaN values given boolean parameter
    if dropnan:
        final_df.dropna(inplace=True)
    return final_df


# Long line to grab data, combine date & time to be the index, and set nan values to recognize '?'
df = pd.read_csv('https://utdallas.box.com/shared/static/7fb4zb0c53hiy500gxazeykpdecer361.txt',sep=';',\
                 parse_dates = {'date' : ['Date', 'Time']}, infer_datetime_format = True, na_values = ['nan','?'],\
                 index_col = 'date')
print(df.head())
print(df.info())

# Replace all NaN values with the mean value for that column
# Might want to do this after split for each set
df = df.fillna(df.mean())
print(df.isnull().sum())

# Graph resampling over the day for sum for each attribute
i = 1
for column in df:
    plt.subplot(7, 1, i)
    df[column].resample('D').sum().plot(color = 'black')
    plt.title(column, y=.5, loc='right')
    i += 1
plt.show()
# Graph Correlation Heat Map between attributes
corr = df.resample('M').sum().corr(method = "spearman")
axHeat1 = plt.axes()
axi1 = sns.heatmap(corr, ax = axHeat1, cmap="BuPu", annot=True)
axHeat1.set_title('Heatmap of Attribute Correlation', fontsize = 24)
plt.show()

'''
Resampling Data over the day for sum
~ this can be changed to see different results
    ~ such as hour, day, month, etc. or sum, mean, etc.
'''
res_df = df.resample('h').sum()
print(res_df.shape)
data_values = res_df.values
# Normalize Attributes
scaler = MinMaxScaler(feature_range=(0, 1))
sc_data = scaler.fit_transform(data_values)
# Reframe data to a Supervised Learning Dataset
rf_data = series_to_supervised(sc_data, 1, 1)
# For this program/project we are predicting the Global Active Power
# So we will drop the last Observation Columns we don't wish to predict
rf_data.drop(rf_data.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(rf_data.head())

'''
Train/Test Split Data
~ Total Time of Data Set is about 4 years
'''
rf_values = rf_data.values
# 80/20 Train/Test Split
train_time = int(rf_values.shape[0] * .8)
train_data = rf_values[:train_time, :]
test_data = rf_values[train_time:, :]
# In past programs we would use sklearn to train/test split.
# But that would shift the rows for better learning in NNs.
# So in this case we are using sequential data so let's keep it constant.
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]
'''
~ LSTM Input must be three-dimensional
~ Input: (Samples, Time Steps, Features)
~ i.e. sample/batch size, size of sequence look-back, number of attributes describing each of the timesteps
'''
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

'''
LSTM Model
'''
# Create Model
lstm_model = krs.Sequential()
lstm_model.add(krs.layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
lstm_model.add(krs.layers.Dropout(.2))
lstm_model.add(krs.layers.LSTM(100, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
lstm_model.add(krs.layers.Dropout(.2))
lstm_model.add(krs.layers.Dense(1))
lstm_model.compile(loss='mae', optimizer='adam')
lstm_model.summary()
# Fit Model
lstm_history = lstm_model.fit(x_train, y_train, epochs=25, batch_size=70, validation_data=(x_test,y_test), shuffle=False, verbose=2)
# Plot Model History
plt.plot(lstm_history.history['loss'], label='Train')
plt.plot(lstm_history.history['val_loss'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Execute Prediction
yh = lstm_model.predict(x_test)
x_test = x_test.reshape((x_test.shape[0], 7))
# Must invert forecast scaling to initial scale
inv_yh = np.concatenate((yh, x_test[:,-6:]), axis=1)
inv_yh = scaler.inverse_transform(inv_yh)
inv_yh = inv_yh[:,0]
# Must invert actual data scaling
y_test = y_test.reshape((len(y_test), 1))
inv_y = np.concatenate((y_test, x_test[:,-6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Calculate Error Values
RMSE = np.sqrt(mean_squared_error(inv_y, inv_yh))
print("Root Mean Squared Error: ", RMSE)
# Plot Actual vs Predicted Graphs
plt.plot(inv_y[:100], label = 'Actual')
plt.plot(inv_yh[:100], label = 'Predicted')
plt.xlabel('Time Steps', fontsize=20)
plt.ylabel('Global Active Power', fontsize=20)
plt.legend()
plt.show()
