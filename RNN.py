'''
Packages:
numpy, pandas, scikit-learn, seaborn, matplotlib, keras, dropbox
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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

# Train/Test Split Data
