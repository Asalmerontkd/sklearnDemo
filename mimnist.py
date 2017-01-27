from sklearn import svm,metrics,datasets
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

digits=datasets.load_digits()
"""
for index,image in enumerate(digits.images[0:9]):
    plt.subplot(3,3,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r)

plt.show()
"""
#print(digits.images[0])
#print("Dimesiones del vector "+str(digits.images.shape))
n_samples=len(digits.images)
images=digits.images.reshape((n_samples, -1))
print(images)
X_train, X_test, y_train, y_test=train_test_split(images,digits.target,test_size=0.1,random_state=42)
#print(X_train[0])
#print(len(X_train))
clf = svm.SVC(C=1.0,kernel='linear')
clf.fit(X_train, y_train)
acc=metrics.accuracy_score(y_test,clf.predict(X_test))
print(acc)
