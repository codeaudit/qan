import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


fin = open('test_minibatch_bs128.log')

ar = []
for i in fin:
    ar.append(i)
	
ar = [float(x) for x in ar]
print(max(ar))

epoch = 40
episode = int(len(ar)/epoch)
print('episode = ' + str(episode))

##baseline VS last episode
t = range(0, epoch)

bl = ar[0:epoch]
lastepisode = []
start = (episode-2) * epoch
for i in xrange(start, start+epoch):
    lastepisode.append(ar[i])
print(bl)
print(lastepisode)
plt.title('Tuning MiniBatch(size=128)')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.plot(t, bl, label='baseline')
plt.plot(t, lastepisode, label='last episode')
plt.legend(loc='lower right')
##

#plt.show()
plt.savefig('vsmini.pdf')



