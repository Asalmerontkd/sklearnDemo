from sklearn import svm
import random
from sklearn import metrics
X=[]
y=[]
for i in range (1000):
    X.append([random.randint(0,100),random.randint(0,100),random.randint(0,100),random.randint(0,100),random.randint(0,100)])
    y.append(0)
    X.append([random.randint(-100,-1),random.randint(-100,-1),random.randint(-100,-1),random.randint(-100,-1),random.randint(-100,-1)])
    y.append(1)

X_train=X[0:900]
y_train=y[0:900]

X_test=X[900:]
y_test=y[900:]

clf = svm.SVC(C=1.0,kernel='linear')
clf.fit(X_train,y_train)
acc=metrics.accuracy_score(y_test,clf.predict(X_test))

print(acc)
