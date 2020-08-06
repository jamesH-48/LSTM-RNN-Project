'''
Packages:
numpy, pandas, scikit-learn, seaborn, matplotlib, keras, dropbox
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
values = df.values

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