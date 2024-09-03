from google.colab import drive # for working in google colab
drive.mount('/content/drive')

import pandas as pd

## upload the dataset csv file from your google drive location
df
df = pd.read_csv("/content/drive/MyDrive/QML_revised/Mag-Nonmag_500.csv")  

df.columns

## Add the most important features needed to train ML models after perfroming feature selection.
features = ['del_Z', 'del_group','d_elect', 'del_electroneg','cell_area',] 
X =df[features]
X.describe()

target = ['mag_state'] ## The target should have the labels as +1 and -1 for classifiction.
y = df[target]

##**** CLASSICAL MACHINE LEARNING ****##

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

X = MinMaxScaler().fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, train_size=0.7, random_state=123)

## Select the required CML you need to train.
#cl_model = SVC(kernel='rbf')      
#cl_model = SVC(kernel = 'linear')
cl_model = RandomForestClassifier()

cl_model.fit(train_X, train_y.values.ravel())
y_pred = cl_model.predict(test_X)
y_pred

## Obtain confusion matrix for CML model.
conf_mat = confusion_matrix(test_y,y_pred, labels=[1,-1]) 
print(conf_mat)

tp, fn, fp, tn = confusion_matrix(test_y,y_pred,labels=[1,-1]).reshape(-1) ## printing the elements of cf_matric i.e., tp, tn, fp, fn.
print('Outcome values : \n', tp, fn, fp, tn)

matrix = classification_report(test_y,y_pred,labels=[1,-1]) ## classification report for precision, recall and f1-score 
print('Classification report : \n',matrix)

##obtaining train and test score for train and test data.
train_score = cl_model.score(train_X, train_y) 
test_score = cl_model.score(test_X, test_y)
print(f"Classical SVC on the training dataset: {train_score:.3f}")
print(f"Classical SVC on the test dataset:     {test_score:.3f}")

##**** QUANTUM MACHINE LEARNING ****##

##Installing necessary modules in google colab.

!pip install qiskit
pip install pylatexenc
pip install qiskit-algorithms
pip install qiskit-machine-learning

##Creating zz feature-map.
from qiskit.circuit.library import ZZFeatureMap

num_features = X.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl",style="iqx",fold=20)

##Defining the Sampler
from qiskit.primitives import Sampler
sampler = Sampler()

##Defining the Quantum Kernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

fidelity = ComputeUncompute(sampler=sampler)

quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

## Defining and training QML model and obtaining scores for teat and train data
from qiskit_machine_learning.algorithms import QSVC

qsvc = QSVC(quantum_kernel=quantum_kernel)

qsvc.fit(train_X, train_y.values.ravel())

qsvc_train_score = qsvc.score(train_X, train_y)
qsvc_test_score = qsvc.score(test_X, test_y)

print(f"QSVC classification train score: {qsvc_train_score:.2f}")
print(f"QSVC classification test score: {qsvc_test_score:.2f}")

## Predicting y using the trained QML model
y_pred = vqc.predict(test_X)
y_pred

#Obtaining cf_matrix and classification metrics for QML model:
conf_mat = confusion_matrix(test_y,y_pred, labels=[1,-1])
print(conf_mat)
tp, fn, fp, tn = confusion_matrix(test_y,y_pred,labels=[1,-1]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(test_y,y_pred,labels=[1,-1])
print('Classification report : \n',matrix)

## cf_matrix heatmap
import seaborn as sns
import matplotlib.pyplot as plt

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True,annot_kws={"size": 20},cbar_kws={'label': 'no. of materials'}, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('True values',fontsize = 15)
ax.set_ylabel('Predicted values', fontsize = 15)
ax.set_title('Stability analysis with QSVC',fontsize = 15);
ax.xaxis.set_ticklabels(['unstable', 'stable'], fontsize = 12)
ax.yaxis.set_ticklabels(['unstable', 'stable'], fontsize =12);

plt.savefig("xxxxxxxxxx.png",dpi = 500)
