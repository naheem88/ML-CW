# ML-CW

# Multi-layer Perceptron Neural Network (MLP-NN)
For this forecasting part of the coursework, you will be working on a specific case study, which is related to exchange rates forecasting. The task of this question is to use a multilayer neural network (MLP-NN) to predict the next step-ahead (i.e. next day) exchange rate of USD/EUR. Daily data (exchangeUSD.xls) have been collected from October 2011 until October 2013 (500 data). The first 400 of them have to be used as training data, while the remaining ones as testing set. Use only the 3rd (i.e. USD/EUR) column from the .xls file, which corresponds to the exchange rates. In this part, the task of the one-step-ahead forecasting of exchange rates will utilise only the “autoregressive” (AR) approach (i.e. time-delayed values of the 3rd column attribute as input variables).

Task Objectives:
In this specific task, utilise only the “autoregressive” (AR) approach, i.e. time-delayed values of the exchange rates (i.e. 3rd column) attribute as input variables. Experiment with various input vectors up to (t-4) level. 

As the order of this AR approach is not known, you need to experiment with various (time-delayed) input vectors and for each case chosen, you need to construct an input/output matrix (I/O) for the MLP training/testing (using “time-delayed” exchange rates)

Each one of these I/O matrices needs to be normalised, as this is a standard procedure especially for this type of NN. 

For the training phase, you need to experiment with various MLP models, utilising these different input vectors and various internal network structures (such as hidden layers, nodes, linear/nonlinear output, activation function, etc.). For each case, the testing performance (i.e. evaluation) of the networks will be calculated using the standard statistical indices (RMSE, MAE, MAPE and sMAPE – symmetric MAPE). 

Finally, provide for your best MLP network, the related results both graphically (your prediction output vs. desired output) and via the stat. indices. In terms of graphics, you can either use a scatter plot or a simple line chart.

Write a code in R Studio to address all the above issues/tasks. 
