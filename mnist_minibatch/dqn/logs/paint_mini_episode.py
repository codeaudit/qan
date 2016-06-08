import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

#fin = open('test_minibatch_bs256.log')
#fin = open('test_lr.log')
fin = open('test_minibatch_bs128.log')

ar = []
for i in fin:
    ar.append(i)

ar = [float(x) for x in ar]
print(max(ar))

epoch = 40
episode = int(len(ar)/epoch)
print('episode = ' + str(episode))

##episode changing
res = []
for i in xrange(0, episode):
    res.append(ar[(i+1)*epoch-1])
print(res)
t = range(0, episode)
plt.title('Tuning MiniBatch(size=128)')
plt.ylabel('Accuracy(%)')
plt.xlabel('Episode')
plt.plot(t, res)
##

#plt.show()
plt.savefig('epmini.pdf')
