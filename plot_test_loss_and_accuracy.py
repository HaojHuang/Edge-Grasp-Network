import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(path,interval=5,select=False):
    data = pd.read_csv(path)
    data = data.iloc[:, :]
    if not select:
        loss = data.loc[:, 'loss'].to_numpy()
        accu = data.loc[:, 'test_accu'].to_numpy()
    else:
        loss = data.loc[:, 'loss'].to_numpy()[::interval]
        accu = data.loc[:, 'test_accu'].to_numpy()[::interval]

    x = np.arange(0, interval * len(loss), interval)
    return loss, accu, x


path1= './edge_grasp_records/test.csv'
loss1, accu1, x1 = read_data(path1, interval=5, select=True)

# path2= './'
# loss2, accu2, x2 = read_data(path2, interval=5, select=True)
#
# path3 = './'
# loss3, accu3, x3 = read_data(path3, interval=5, select=True)
#
# path4 = './'
# loss4, accu4, x4 = read_data(path4, interval=5, select=True)


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,5))


line1a, = ax1.plot(x1,loss1,color='r',lw=3)
line1b, = ax2.plot(x1,accu1,color='r',lw=3)

# line2a, = ax1.plot(x2,loss2,color='g',lw=3)
# line2b, = ax2.plot(x2,accu2,color='g',lw=3)

# line3a, = ax1.plot(x3,loss3,color='g',lw=3)
# line3b, = ax2.plot(x3,accu3,color='g',lw=3)
#
# line4a, = ax1.plot(x4,loss4,color='k',lw=3)
# line4b, = ax2.plot(x4,accu4,color='k',lw=3)

title1 = 'Test Loss v.s. Epoch'
ax1.set_title(title1,size=18)
title2 = 'Test Accuracy v.s. Epoch'
ax2.set_title(title2,size=18)

ax1.set_xlim(0,201)
ax2.set_xlim(0,201)

ax1.set_xticks(np.arange(0, 201, 50))
ax1.set_yticks(np.linspace(0.2, 0.8, 7))
ax2.set_xticks(np.arange(0, 201, 50))
ax2.set_yticks(np.linspace(0.5, 1, 6))

ax1.set_xlabel('Epoch',size=16,weight='roman')
ax1.set_ylabel('Test Loss',size=16,weight='roman')

ax2.set_xlabel('Epoch',size=16,weight='roman')
ax2.set_ylabel('Test Accuracy',size=16,weight='roman')

ax1.legend([line1a,],['EdgeGraspNet', ])
ax2.legend([line1b,],['EdgeGraspNet', ])

ax1.grid(True, linestyle='-', color=[0.8, 0.8, 0.8])
ax2.grid(True, linestyle='-', color=[0.8, 0.8, 0.8])
plt.show()