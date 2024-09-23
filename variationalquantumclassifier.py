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

##Creating Real amplitude Ansatz
from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="iqx",fold=20)

## Defining Optimizer and Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
optimizer = COBYLA(maxiter=200)
sampler = Sampler()

##Creating a function for creating call back graph from model optimization
from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Mag-NonMag classification with VQC Obj.fxn.value vs iteration ")
    plt.xlabel("Iteration",fontsize=20)
    plt.ylabel("Objective function value",fontsize=20)
    plt.plot(range(len(objective_func_vals)), objective_func_vals)

    plt.savefig('xxxxxx.png', dpi=300, bbox_inches='tight')
    plt.show()

##Defining the VQC model and training the model:
import time
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
    )

objective_func_vals = []

start = time.time()
vqc.fit(train_X, train_y.values.ravel())
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

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

train_score_q4 = vqc.score(train_X, train_y)
test_score_q4 = vqc.score(test_X, test_y)

## Obtain the final scores for train and test data.
print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")
