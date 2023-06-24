import numpy as np
from ZIFA import ZIFA
from ZIFA import block_ZIFA


# This gives an example for how to read in a real data called input.table. 
# genes are columns, samples are rows, each number is separated by a space. 
# If you do not want to install pandas, you can also use np.loadtxt: https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html

dataname = r'logdata/logyan.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
labelname = r'label//yan.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(np.int)
n = y.max() - y.min() +1

idx = np.argwhere(np.all(x[..., :] == 0, axis=0))
a2 = np.delete(x, idx, axis=1)
Z, model_params = block_ZIFA.fitModel(a2, 5)

from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
clf = KMeans(n_clusters=n)
clustering = clf.fit(Z)
yp = clustering.predict(Z)

np.savetxt('Z-yan.csv', Z, fmt='%f', delimiter=',', encoding='utf-8')
np.savetxt('yan.csv', yp, fmt='%d', delimiter=',', encoding='utf-8')

from sklearn import metrics
print('ARI:', metrics.adjusted_rand_score(y, yp))