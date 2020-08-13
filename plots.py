import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


predictions = pd.read_csv("data/predictions.csv", header=None)
print(predictions.shape)
y = pd.read_csv("data/y_AR.csv", header=None)
y_test = y[-1914:]
print(y_test.head())
y_test = y_test[0].tolist()
predictions = predictions[0].tolist()

axes = plt.gca()


axes.set_ylim([0.4,1.3])

plt.plot(predictions, 'r--', linewidth=1)
plt.plot(y_test, 'g')
plt.show()


