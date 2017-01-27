from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib

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
lista=convierte_a_listas("Que tal esta es una lista para el mercado algunas de las cosas que quiero comprar es jamon queso pechuga de pavo y cereal")
print(lista)
features=prepara_frase(lista)
print(features)
v = DictVectorizer(sparse=False)
#X = v.fit_transform(features)
v.fit(features)
joblib.dump(v, 'vectorizer.pkl')
