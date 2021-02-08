import numpy as np
import matplotlib.pyplot as plt

RNN_data =np.array([0.7494,
        0.7496,
        0.6291,
        0.8965,
        0.3949,
        0.4685,
        0.5732,
        0.4544,
        0.9935,
        0.3648,
        1.1323,
        0.5867,
        0.4170,
        1.1396,
        0.7958,
        1.1698,
        1.0329,
        0.4488,
        0.3495,
        0.3690])
LSTM_data = np.array([
    0.6814,
 0.6803,
0.7227,
0.6716,
0.6782,
  0.6440,
 0.7632,
  0.7714,
 0.8512,
 0.6447,
 0.7259,
 0.6827,
 0.5281,
 0.8415,
 0.7501,
 1.0589,
 0.4456,
 0.6052,
 0.5902,
 0.9635
])
plt.plot(RNN_data) # plotting by columns
plt.plot(LSTM_data) # plotting by columns

plt.show()
