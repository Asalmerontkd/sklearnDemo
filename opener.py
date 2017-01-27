import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn import svm,metrics
from sklearn.cross_validation import train_test_split


with open('corpus.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
def prepara_frase(data):
    features=[]
    feature={}
    targets=[]

    for i,vector in enumerate(data):

        if vector[0]!='-' and  vector[0]!='':

            feature['0']=str(data[i-2][1]).lower()
            feature['1']=str(data[i-1][1]).lower()
            feature['2']=str(data[i][1]).lower()
            feature['3']=str(data[i+1][1]).lower()
            feature['4']=str(data[i+2][1]).lower()
            features.append(feature)
            #print feature
            feature={}
            #if vector[0]!='af':
            targets.append(vector[0][0])
            #else:
                #targets.append('*')
    return features,targets

features,target=prepara_frase(data)
for i,element in enumerate(target):
    if element=='t' or element=='a':
        target[i]='*'
v = DictVectorizer(sparse=False)
v.fit(features)
joblib.dump(v, 'vectorizer.pkl')
transformed=v.transform(features)
print(transformed[0])

X_train, X_test, y_train, y_test=train_test_split(transformed,target,test_size=0.1,random_state=42)
#print(X_train[0])
#print(len(X_train))
clf = svm.SVC(C=1,kernel='linear')
clf.fit(X_train, y_train)
acc=metrics.accuracy_score(y_test,clf.predict(X_test))
joblib.dump(clf, 'clasifier.pkl')
print(acc)
