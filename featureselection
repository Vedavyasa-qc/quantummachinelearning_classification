from google.colab import drive # for working in google colab
drive.mount('/content/drive')

import pandas as pd

## upload the dataset csv file from your google drive location
df
df = pd.read_csv("/content/drive/MyDrive/QML_revised/h_form_1000.csv")  

df.columns

####******* RANDOM FOREST FEATURE IMPORTANCE *******####

features = ['id', 'd_elect', 'mean_Z', 'del_Z', 'mode_Z', 'L2_norm', 'L3_norm', ## Add all thecolumn names into features except containing labels.
       'mean_group', 'del_group', 'mode_group', 'mean_period', 'del_period',
       'mode_period', 'mean_val', 'del_val', 'mode_val', 'mean_electroneg',
       'del_electroneg', 'mode_electroneg', 'entropy', 'cell_area', 'n_metal',
       'mag_state']
y = df['label']
#Load X Variables into a Pandas Dataframe with columns
X = df.drop(['id','label','hform'], axis = 1)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.3)
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train,y_train.values.ravel())

##Plotting a bargraph showing the features based on their relative importance.
f_i = list(zip(features,rf.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.xlabel("Relative Importance",fontsize=15)
plt.ylabel("Features",fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title("Feature Importance",fontsize=15)

plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

plt.savefig("xxxxxxx.png",dpi = 500)

####******* PEARSON CORRELATION HEATMAP *******####

y = df['mag_state']
#Load X Variables into a Pandas Dataframe with columns
X = df.drop(['id'], axis = 1)

import seaborn as sns
cor = X_train.corr()
plt.figure(figsize=(10,10))
fig=sns.heatmap(cor, cmap=plt.cm.CMRmap_r,annot=True,annot_kws={"size": 8},cbar_kws={ 'orientation': 'horizontal'})
plt.yticks(rotation=0)
fig.set_xticklabels(fig.get_xmajorticklabels(), fontsize = 10)
fig.set_yticklabels(fig.get_ymajorticklabels(), fontsize =10)

plt.savefig("xxxxxxxx.png",dpi = 500)

plt.show()
