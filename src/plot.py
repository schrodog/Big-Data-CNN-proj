import matplotlib.pyplot as plt
import numpy as np

className = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
accuracy = [0.9669, 0.9802, 0.9456, 0.9217, 0.9574, 0.9423, 0.9678, 0.9717, 0.9746, 0.9776]
recall = [0.874, 0.923, 0.73,  0.648, 0.759, 0.72,  0.85,  0.828, 0.835, 0.862]
precision = [0.81000927, 0.88409962, 0.72709163, 0.60055607, 0.80402542, 0.7079646, 0.83170254, 0.88178914, 0.90367965, 0.9092827 ]
f1_score = [0.84078884, 0.90313112, 0.72854291, 0.62337662, 0.7808642, 0.71393158,0.84075173, 0.85404848, 0.86798337, 0.88501027]
mcc = [0.08512222, 0.09095556, 0.06995556, 0.06001111, 0.07384444, 0.0687,
 0.08308889, 0.08156667, 0.08251111, 0.08524444]


x = np.arange(0.0, 10.0)
# x = className
w = 0.2
plt.bar(x-w*3/2, accuracy, width=0.2, align='center', label='accuracy')
plt.bar(x-w/2, recall,  width=w, align='center', label='recall', tick_label=className)
plt.bar(x+w/2, precision, width=w, align='center', label='precision', tick_label=className)
plt.bar(x+w*3/2, f1_score,width=w, align='center', label='f1 score')
plt.legend()
plt.show()

plt.title('Matthew Correlation Coefficient')
plt.bar(className, mcc ,width=0.5, align='center')
plt.show()



