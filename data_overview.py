#%%
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data=pd.read_csv('dataset1.csv')
data

# %%

#quick overview of the size distribuion of the test packages
data.boxplot(column='size')

# %%

stat=data[['AtoB','BtoC','CtoD']].describe()


# KDE plot to see the distribution of observations between point.
# using bandwith of 0.05 to smooth the noise.
# skip BtoC segment for better visualisation.


fig = plt.figure()
ax1 = fig.add_subplot(121)
font_size = 14
bbox = [0, 0, 1, 1]
ax1.axis('off')
stat_table = ax1.table(cellText=stat.values, rowLabels=stat.index, bbox=bbox, colLabels=stat.columns)
stat_table.set_fontsize(20)

ax2 = fig.add_subplot(122)
data['AtoB'].plot.kde(bw_method=0.05, color='blue', label='AtoB')
data['CtoD'].plot.kde(bw_method=0.05, color='red', label='CtoD')
plt.xlim(0,0.12)
plt.tight_layout()
plt.legend()

plt.show()

#%%
data['BtoC']=data['AtoC']-data['AtoB']
data['CtoD']=data['AtoD']-data['AtoC']

data_naAB=data[data['AtoB'].isnull()]
data_naBC=data[data['BtoC'].isnull()]
data_naCD=data[data['CtoD'].isnull()]
data_naAB=data_naAB[['time', 'AtoB']]
data_naBC=data_naBC[['time', 'BtoC']]
data_naCD=data_naCD[['time', 'CtoD']]
data_naAB.fillna(0.13, inplace=True)
data_naBC.fillna(0.132, inplace=True)
data_naCD.fillna(0.134, inplace=True)
plt.plot(data['time'], data['AtoB'], color='blue', label='AtoB')
plt.plot(data['time'], data['BtoC'], color='green', label='BtoC')
plt.plot(data['time'], data['CtoD'], color='red', label='CtoD')
plt.plot(data_naAB['time'], data_naAB['AtoB'], linestyle='None', marker="o", markerfacecolor="blue")
plt.plot(data_naBC['time'], data_naBC['BtoC'], linestyle='None', marker="o", markerfacecolor="green")
plt.plot(data_naCD['time'], data_naCD['CtoD'], linestyle='None', marker="o", markerfacecolor="red")
plt.xticks(np.arange(0, 100, 5))
plt.legend()
plt.title('Loss of individual test packages over time (in sec) from three segments')
plt.show()

# %%
data['BtoC']=data['AtoC']-data['AtoB']
data['CtoD']=data['AtoD']-data['AtoC']
data['AtoB'].fillna(0.13, inplace=True)
data['BtoC'].fillna(0.13, inplace=True)
data['CtoD'].fillna(0.13, inplace=True)
plt.figure(figsize=(15, 4))
plt.plot(data['time'], data['AtoB'], marker="o", markerfacecolor="blue",label='AtoB')
plt.plot(data['time'], data['BtoC'], marker="o", markerfacecolor="green", label='BtoC')
plt.plot(data['time'], data['CtoD'], marker="o", markerfacecolor="red", label='AtoB')
plt.xticks(np.arange(0, 100, 5))

plt.yticks([])
plt.legend()
plt.show()

# %%
# generate new data points: probability & cumulative probability distributions for segments
counts,bin_edges=np.histogram(data['AtoB'],bins=500,density=True)

data['PDF_AtoB']=counts/sum(counts)
data['CDF_AtoB']=data['PDF_AtoB'].cumsum()


#data['PDF_BtoC']=data['BtoC']/float(len(data))
#data['CDF_BtoC']=data['PDF_BtoC'].cumsum()
#data['PDF_CtoD']=data['CtoD']/float(len(data))
#data['CDF_CtoD']=data['PDF_CtoD'].cumsum()


# %%

plt.plot(data['time'],data['CDF_AtoB'])

plt.show()

# %%

#Likelihood of occurance of each delay/loss.
sns.kdeplot(data = data['AtoB'], cumulative = True, label = 'AtoB')
sns.kdeplot(data = data['BtoC'], cumulative = True, label = 'BtoC')
sns.kdeplot(data = data['CtoD'], cumulative = True, label = 'CtoD')
plt.legend()
plt.xticks(np.arange(-0.01, 0.14, 0.01))
plt.yticks(np.arange(0, 1.0, 0.1))
plt.grid()
plt.show()


#%%

def cdf(df, col_name, cdf_col):
    counts,bin_edges=np.histogram(df[col_name],bins=500,density=True)
    df[cdf_col]=(counts/sum(counts)).cumsum()
    return df

data.sort_values('AtoB', inplace=True)
data=cdf(data,'AtoB','CDF_AtoB')

data.sort_values('BtoC', inplace=True)
data=cdf(data,'BtoC','CDF_BtoC')

data.sort_values('CtoD', inplace=True)
data=cdf(data,'CtoD','CDF_CtoD')


plt.plot(data['AtoB'],data['CDF_AtoB'], color='blue', label='AtoB')
#plt.plot(data['BtoC'],data['CDF_BtoC'], color='green', label='BtoC')
plt.plot(data['CtoD'],data['CDF_CtoD'], color='red', label='CtoD')

#plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(0, 1.0, 0.1))
plt.legend()
plt.title('Cumulative Probability Distribution')
plt.show()





# %%
