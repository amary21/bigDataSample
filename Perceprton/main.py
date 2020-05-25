import perceptron
import csvconv

#inputs 
lr=float(input("Input Learning Rate:"))
Epochs=int(input("Input Number Of Epochs:"))

#training
floatdata=csvconv.tofloat('train.csv')
weights=perceptron.train(floatdata,lr,Epochs)
print("Weights\n ",weights)

#testing
floattest=csvconv.tofloat('test.csv')
score=perceptron.test(floattest,weights)
print('Score=',score,'/',len(floattest))
acc=(score/len(floattest))*100
print("Accuracy=",acc)