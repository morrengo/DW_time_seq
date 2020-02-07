import cv2 
import numpy as np
from matplotlib import pyplot as plt
import numpy
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', action='store', dest='plot_location' , help='plot_location')

parser.add_argument('-c', action='store', default=10,
                    dest='clust_num',
                    help='how many clusters')

parser.add_argument('-a', action='store', default=10,
                    dest='avg_num',
                    help='how many times should plot and derivative be averaged')


results = parser.parse_args()
CLUST_NUM = int(results.clust_num)
PLOT_LOCATION = results.plot_location
AVG_NUM = int(results.avg_num)

def smooth(x,window_len=20):
    if window_len<3:
        return x
    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=eval('numpy.'+'hanning'+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def average(x):
	y = np.zeros(len(x))
	for i in range(0,len(x)-1):
		y[i] = (x[i-1]+x[i+1])/2
	y[0] = x[0]
	y[len(x)-1] = x[len(x)-1]
	return y


img = cv2.imread(PLOT_LOCATION)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img < 100
height = img.shape[0]
width = img.shape[1]
time_seq = np.zeros(width)
for i in range(0,width):
	for j in range(height-1,-1,-1):
		if(img[j][i] != 0):
			time_seq[i] = height-j
			break
orig_seq = time_seq
time_seq = smooth(time_seq)

for i in range(0,AVG_NUM):
	time_seq = average(time_seq)
DERIVATIVE_RANGE = 20
derivative = np.zeros(width)
for i in range(0,width - DERIVATIVE_RANGE):
	derivative[i] = (time_seq[i+DERIVATIVE_RANGE] - time_seq[i])/DERIVATIVE_RANGE
derivative = derivative/(derivative.max()-derivative.min())*int(CLUST_NUM/2)

for i in range(0,3*AVG_NUM):
	derivative = average(derivative)

plt.subplot(121)
plt.plot(time_seq)
plt.subplot(122)
plt.plot(derivative)
plt.show()

plt.plot(orig_seq)
clusters = []
for i in range(-int(CLUST_NUM/2),int(CLUST_NUM/2)):
	clusters.append([x for x in range(0,len(derivative)) if derivative[x] > i-0.5 and derivative[x] <= i+0.5])
clusters_plots = np.zeros(shape=(len(clusters),width))

for i in range(0,len(clusters)):
	for j in range(0,len(time_seq)):
		if(j in clusters[i]):
			clusters_plots[i][j] = orig_seq[j]
	plt.plot(clusters_plots[i],label = str(i-int(CLUST_NUM/2)))
plt.legend()
plt.show()
