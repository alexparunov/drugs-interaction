# Identifying Drug-Drug Interactions and Drug Name Entity Recognition using SVM and CRF

## Abstract

Adverse drug-drug interactions (DDIs) remain a leading cause of morbidity and mortality around the world. Computational approaches that leverage the available big data to identify potential DDIs can greatly facilitate pre-market targeted clinical drug safety testings as well as post-market drug safety surveillance. However, most existing computational approaches only focus on binary prediction of the occurrence of DDIs, i.e., whether there is an existing DDI or not. Prediction of the actual DDI types will help us understand specific DDI mechanism and identify proper early prevention strategies. In addition, the nonlinearity and heterogeneity in drug features are rarely explored in most existing works. In this paper, we propose a deep learning model to predict fine-grained DDI types. It captures nonlinearity with nonlinear model structures and through layers of abstraction, and resolves the structural heterogeneity via projecting different subsequences of fingerprint features to the same hidden states that indicate a particular interaction pattern. Moreover, we proposed a multi-task deep model to perform simultaneous prediction for a focused task (i.e. a DDI type) along with its auxiliary tasks, thus to exploit the relatedness among multiple DDI types and improve the prediction performance. Experimental results demonstrated the effectiveness and usefulness of the proposed model. 

#### Setup:
```
bash setup.sh
```

#### Usage(from root directory):
```
usage: main.py [-h] [-p] [-t TASK] [--train] [--test] [-f FOLDER_INDEX]
               [-i MODEL_INDEX] [-r RATIO] [-c CLASSIFIER]

Train or Test model

optional arguments:
  -h, --help            show this help message and exit
  -p, --parse           Parse files (Should be done for first run, later it's optional)
  -t TASK, --task TASK  Task of problem. 1 - NER task, 2 - DDI task.
  --train               Train model
  --test                Test model at index i
  -f FOLDER_INDEX, --folder_index FOLDER_INDEX
                        Folder number. 1 - drugbank, 2 - medline
  -i MODEL_INDEX, --model_index MODEL_INDEX
                        Index of a model to test
  -r RATIO, --ratio RATIO
                        Ratio of data to use for training
  -c CLASSIFIER, --classifier CLASSIFIER
                        Classifier to use. 1 - SVM, 2 - CRF
```

#### Best Results using Golden Standard of competition:
```
TASK 1: NER Drugbank (SVM with linear kernel) - Precision = 0.76, Recall = 0.55, F-Score = 0.64
TASK 1: NER Drugbank (CRF using LBFGS algorithm) - Precision = 0.8, Recall = 0.56, F-Score = 0.66
TASK 1: NER Medline (SVM with liner kernel) - Precision = 0.56, Recall = 0.23, F-Score = 0.33
TASK 1: NER Medline (CRF using LBFGS algorithm) - Precision = 0.55, Recall = 0.18, F-Score = 0.27

TASK 2: DDI Drugbank (SVM with linear kernel) - Precision = 0.42, Recall = 0.21, F-Score = 0.28
TASK 2: DDI Drugbank (CRF using LBFGS algorithm) - Precision = 0.40, Recall = 0.21, F-Score = 0.28
TASK 2: DDI Medline (SVM with linear kernel) - Precision = 0.34, Recall = 0.22, F-Score = 0.27
TASK 2: DDI Medline (SVM with linear kernel) - Precision = 0.32, Recall = 0.18, F-Score = 0.23
```

#### Future developments
Since results of first NER task are quite impressive even on official competition's scoreboard, I might move forward and implement CNN for TASK 2. After reading many papers, Convolutional Neural Net shows better results than SVM or CRF.

More information about the competition and tasks can be found on official website of this 2013 semeval contest: https://www.cs.york.ac.uk/semeval-2013/task9.html 
