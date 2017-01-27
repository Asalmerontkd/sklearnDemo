from sklearn.externals import joblib
from sklearn import svm,metrics



def convierte_a_listas(oracion):
	wap=oracion.split(" ")
	words=['-','-']
	for i in wap:
		words.append(i)
	words=words+['-','-']
	return words
def prepara_frase(words):
	features=[]
	feature={}
	for i in range(len(words[2:-2])):
		i=i+2
		feature['0']=str(words[i-2]).lower()
		feature['1']=str(words[i-1]).lower()
		feature['2']=str(words[i]).lower()
		feature['3']=str(words[i+1]).lower()
		feature['4']=str(words[i+2]).lower()
		features.append(feature)
		feature={}
	return features
def giveTags(oracion):
	vectorizer=joblib.load('vectorizer.pkl')
	clasifier=joblib.load('clasifier.pkl')
	lista=convierte_a_listas(oracion)
	features=prepara_frase(lista)
	vec=vectorizer.transform(features)
	pred=clasifier.predict(vec)
	return pred,lista
