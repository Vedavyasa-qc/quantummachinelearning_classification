**Description of the files in the Repository**:

**variationalquantumclassifier**: This file contains code for training QML models using Variational Quantum Classifier (VQC) algorithm

**quantumsupportvectorclassifier**: This file contains code for training QML models using Quantum Support Vector Classifier (QSVC) algorithm

**quantumneuralnetworks**: This file contains code for training QML models using Quantum Neural Networks Classifier (QNN) algorithm

**featureselection**: This file contains codefor performing feature selction using Random Forest Feature importance (RFF) and Pearson Correlation Heatmap (PCH).

**NecessaryPackages**:

For CML, all the necessary packges used like pandas, matplotlib, sklearn interfaces etc. are directly imported into colab notebook using 'import' command provided in the respective codes.

For QML, following packages are installed seperately before creating and training the QML models. The usual 'pip' dependency used for installing regular packages  will serve the purpose. However, the code for installation is given in the files inside the repository for clarity.

1. qiskit - version 1.2.0
2. pylatexenc - version 2.10
3. qiskit-algorithms - version 0.3.0
4. qiskit-machine-learning - version 0.7.2

**Instructions for reproducing the Results**:

The CML models are trained one by one in each kind of classification performed in this study and the performance is noted before proceeding on to create QML models and train them. The users should carefully distinguish the CML and QML models in the codes. Annotations are added in the codes as headings to briefly explain the purpose and functioning of the piece of the code being used. The three QML files shoud be seperately seen as individual training models where different algorithms are tested and finally the results are compared for further study.  

**Note**:

The 3 csv files contains the datasets for performing three types of classifications i.e., classification of TMCs and TMOs, Stability Analysis and Magnetic nature Analysis with 350 materials, 1000 materials and 500 materials respectively.
