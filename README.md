# air-ratio-model
This is my implementation of an Extreme Learning Machine used to predict the future air-ratio (a metric that describes the difference between the actual air-fuel ratio in the engine and the desired stoichiometric air-fuel ratio. Based off of this paper: https://link.springer.com/article/10.1007/s00521-014-1555-7#Abs1. My intention is to use this to improve the fuel efficiency of the Brum Eco Racing car, as part of the Shell Eco Marathon.

See below for some test results:
![test results - predicted vs actual air ratio](https://i.imgur.com/RO0z2tA.png)
![close up of the above image](https://i.imgur.com/VSNH1d2.png)
N.B. The first graph shows the results for a set of ~ 2000 test samples, the second is a close-up view of the first.
The green line is the actual air ratio and the red line is the air ratio as predicted by my model. These results were obtained after training the model on ~ 3900 training samples
The input data consists of the engine speed (rpm), throttle position (degrees) and air-ratio for the previous 2 time steps.

From the graph we can see that the model predicts the air ratio with a good degree of accuracy, however, It tends to overshoot the actual values. I aim to improve upon this by revising certain parts of the model and by experimenting with different kinds of training data.
