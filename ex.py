import matplotlib.pyplot as plt

batchsize = ['1500','2000','2500']

accnb1 = [0.8253333333333334,0.826,0.8266666666666667]
accnb2 = [0.43933333333333335 ,0.44133333333333336 ,0.44266666666666665]

plt.bar(batchsize,accnb1, width = 0.4)
plt.xlabel("batch size")
plt.ylabel("Acc naive bayes")
plt.show()

plt.bar(batchsize,accnb2,width = 0.4)
plt.xlabel("batch size")
plt.ylabel("Acc naive bayes")
plt.show()


accsvm1 = [0.8813333333333333 , 0.8346666666666667 ,0.8333333333333334 ]
accsvm2 = [ 0.6426666666666667 , 0.646 ,0.646 ]

plt.bar(batchsize,accsvm1,width = 0.4)
plt.xlabel("batch size")
plt.ylabel("Acc svm")
plt.show()


plt.bar(batchsize,accsvm2,width = 0.4)
plt.xlabel("batch size")
plt.ylabel("Acc svm")
plt.show()


acclogreg1 = [0.9786666666666667 , 0.8326666666666667 , 0.8266666666666667 ]
acclogreg2 = [0.8446666666666667 ,0.6446666666666667 ,0.7633333333333333 ]


plt.bar(batchsize,acclogreg1,width = 0.4)
plt.xlabel("batch size")
plt.ylabel("Acc log reg")
plt.show()


plt.bar(batchsize,acclogreg2,width = 0.4)
plt.xlabel("batch size")
plt.ylabel("Acc log reg")
plt.show()
