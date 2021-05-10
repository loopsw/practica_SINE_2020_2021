'''
El script no contiene acentos para evitar problemas de codificacion.

Es importante tener el entorno de desarrollo configurado para utf-8 
o puede dar problemas al leer ciertos caracteres del dataset de DETOXIS

@author: Jackson Reyes
'''

# importación de liberias
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from unicodedata import normalize
import re
import nltk

# nltk.download('stopwords') # Para descargar las stopwords desde nltk
from nltk.corpus import stopwords

# Elimina los acentos para evitar problemas de codificación
def limpiaAcentos(string): 
    # -> NFD y eliminar diacríticos
    s = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
            normalize( "NFD", string), 0, re.I
        )

    # -> NFC
    return normalize( 'NFC', s)

#Funcion que dado un documento de Spacy devuelve una lista de palabras.
def getToken(docSpacy):
    listTokens = []
    for token in docSpacy:
        listTokens.append(token.text)
    
    return listTokens

# los tokens solo son aquellos que no esten en stopwords
def getToken2(docSpacy,stopwords):
    listTokens = []
    for token in docSpacy:
        if token.text not in stopwords:
            listTokens.append(token.text)
    
    return listTokens

# los tokens son aquellos que no estan en stopwords y todos convertidos a minuscula
def getToken3(docSpacy,stopwords):
    listTokens = []
    for token in docSpacy:
        if token.text.lower() not in stopwords:
            listTokens.append(token.text.lower())
    
    return listTokens

# elimina simbolos
def getToken4(docSpacy,stopwords):
    listTokens = []
    for token in docSpacy:         
        token = re.sub('[^a-zA-Z]', ' ', token.text)
        token = token.lower()
        if token not in stopwords:
            listTokens.append(token)
    
    return listTokens

# limpieza de acentos
def getToken5(docSpacy,stopwords):
    listTokens = []
    docSpacy = limpiaAcentos(str(docSpacy))
    docSpacy = re.sub('[^a-zA-Z]', ' ', str(docSpacy))
    docSpacy = docSpacy.lower()
    docSpacy = docSpacy.split()
    for token in docSpacy:   
        if token not in stopwords:
            listTokens.append(token)
    
    return listTokens

# limpieza de espacios en blanco y seleccionado lemas
def getToken6(docSpacy,stopWords,nlp):
    listTokens = set()
    docSpacy = limpiaAcentos(str(docSpacy).strip())
    docSpacy = re.sub('[^a-zA-Z]', ' ', str(docSpacy))
    docSpacy = docSpacy.lower().strip()
    docSpacy = nlp(docSpacy)
    for token in docSpacy:   
        if token.text not in stopWords and token.text.strip() != '':
            listTokens.add(token.lemma_)
    
    return listTokens

#Funcion tonta que devuelve el documento pasado por parametro.        
def dummy(doc):
    return doc

#Funcion principal para classificar comentarios en base a su toxicidad
def processDetoxisTraining(path):
    #Leemos el archivo csv
    print("Leyendo detoxis dataset")
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')
    print(dataset.shape)

    #Cargamos el modelo del lenguaje
    nlp= spacy.load('es_core_news_sm')     
    
    #Procesamos los comentarios
    print("Procesando comentarios para extraer sus palabras")
    lstDocsDetoxis = []
    lstObjetiveClass = []
    for index, row in dataset.iterrows():
        print("Generando documento detoxis "+ str(index))
        getToken(nlp(row["comment"]))
        lstDocsDetoxis.append(getToken(nlp(row["comment"])))
        lstObjetiveClass.append(row["toxicity"])
    
    #Generamos la matriz td-idf
    print("Generando matriz de tf-idf")
    tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    tfidf_matrix = tfidf.fit_transform(lstDocsDetoxis)
    lstFeaturesTFIDFByComment = tfidf_matrix.toarray()
    
    #Configuramos el modelo de SVM para entrenar/predecir
    print("Entrenando el algoritmo SVM")
    clf = svm.SVC(kernel='linear') 
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy
    print("Calculando accuaracy")
    #scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10, n_jobs=4) 
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=2, n_jobs = 4) 
    
    print("Calculando Cross-Fold Validation")
    classNames = dataset.toxicity.unique() 
    #classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 10)
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 2,n_jobs= 4)  
    clasificationReport = classification_report(lstObjetiveClass, classesPredicted, target_names=classNames) 
    matrixConfusion = confusion_matrix(lstObjetiveClass, classesPredicted)
    
    
    print("RESULTADOS")
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    print("\nMostrando metricas mas precisas y matriz de confusion")
    print(clasificationReport)    
    print("Matriz de confusión con todo los datos")
    print(matrixConfusion)

    
# apartado 1: Función modificada del modelo basico para eliminar las stopwords
def primerAjuste(path):
    #Leemos el archivo csv
    print("Leyendo detoxis dataset")
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')
    print(dataset.shape)

    #Cargamos el modelo del lenguaje
    nlp= spacy.load('es_core_news_sm')     
    
    #Procesamos los comentarios
    print("Procesando comentarios para extraer sus palabras")    
    stopWords = set(stopwords.words('spanish'))
    lstDocsDetoxis = []
    lstObjetiveClass = []
    for index, row in dataset.iterrows():
        #print("Generando documento detoxis "+ str(index))
        #getToken(nlp(row["comment"]))
        #lstDocsDetoxis.append(getToken2(nlp(row["comment"]),stopWords))
        #lstDocsDetoxis.append(getToken3(nlp(row["comment"]),stopWords))
        #lstDocsDetoxis.append(getToken4(nlp(row["comment"]),stopWords))
        lstDocsDetoxis.append(getToken5(nlp(row["comment"]),stopWords))
        lstObjetiveClass.append(row["toxicity"])
    
    #Generamos la matriz td-idf
    print("Generando matriz de tf-idf")
    tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    tfidf_matrix = tfidf.fit_transform(lstDocsDetoxis)
    lstFeaturesTFIDFByComment = tfidf_matrix.toarray()
    
    #Configuramos el modelo de SVM para entrenar/predecir
    print("Entrenando el algoritmo SVM")
    clf = svm.SVC(kernel='linear') 
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy
    print("Calculando accuaracy")
    #scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10,n_jobs=4)
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=2,n_jobs=4) 
    
    print("Mostrando metricas mas precisas y matriz de confusion")
    classNames = dataset.toxicity.unique() 
    #classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 10,n_jobs=4)
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 2,n_jobs=4)       
    clasificationReport = classification_report(lstObjetiveClass, classesPredicted, target_names=classNames)
    matrixConfusion = confusion_matrix(lstObjetiveClass, classesPredicted)
    
    print("RESULTADOS")
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    print("\nMostrando metricas mas precisas y matriz de confusion")
    print(clasificationReport)    
    print("Matriz de confusión con todo los datos")
    print(matrixConfusion)


# Apartado 2: Función modificada del modelo basico, seleccionando lemmas en lugar de palabras
def segundoAjuste(path):
    #Leemos el archivo csv
    print("Leyendo detoxis dataset")
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')
    print(dataset.shape)

    #Cargamos el modelo del lenguaje
    nlp= spacy.load('es_core_news_sm')     
    
    #Procesamos los comentarios
    print("Procesando comentarios para extraer sus palabras")    
    stopWords = set(stopwords.words('spanish'))
    lstDocsDetoxis = []
    lstObjetiveClass = []
    for index in range(0,len(dataset)):
        lstDocsDetoxis.append(getToken6(dataset["comment"][index],stopWords,nlp))
        lstObjetiveClass.append(dataset["toxicity"][index])
    
    #Generamos la matriz td-idf
    print("Generando matriz de tf-idf")
    tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    tfidf_matrix = tfidf.fit_transform(lstDocsDetoxis)
    lstFeaturesTFIDFByComment = tfidf_matrix.toarray()
    
    #Configuramos el modelo de SVM para entrenar/predecir
    print("Entrenando el algoritmo SVM")
    clf = svm.SVC(kernel='linear') 
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy
    print("Calculando accuracy")
    #scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10,n_jobs=4)
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=2,n_jobs=4) 

    print("Mostrando metricas mas precisas y matriz de confusion")
    classNames = dataset.toxicity.unique() 
    #classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 10,n_jobs=4)       
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 2,n_jobs=4)       
    clasificationReport = classification_report(lstObjetiveClass, classesPredicted, target_names=classNames)
    matrixConfusion = confusion_matrix(lstObjetiveClass, classesPredicted)
    
    print("RESULTADOS")
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    print("\nMostrando metricas mas precisas y matriz de confusion")
    print(clasificationReport)    
    print("Matriz de confusión con todo los datos")
    print(matrixConfusion)

    
    
# Apartado 3: Función modificada del modelo basico para añadir caracteristicas al clasificador
def tercerAjuste(path):
    #Leemos el archivo csv
    print("Leyendo detoxis dataset")
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')
    print(dataset.shape)

    #Cargamos el modelo del lenguaje
    nlp= spacy.load('es_core_news_sm')     
    
    #Procesamos los comentarios
    print("Procesando comentarios para extraer sus palabras")    
    stopWords = set(stopwords.words('spanish'))
    lstDocsDetoxis = []
    lstObjetiveClass = []
    for index in range(0,len(dataset)):
        lstDocsDetoxis.append(getToken6(dataset["comment"][index],stopWords,nlp))
        lstObjetiveClass.append(dataset["toxicity"][index])
    
    #Generamos la matriz td-idf
    print("Generando matriz de tf-idf")
    tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    tfidf_matrix = tfidf.fit_transform(lstDocsDetoxis)
    lstFeaturesTFIDFByComment = tfidf_matrix.toarray()
    
    #Configuramos el modelo de SVM para entrenar/predecir
    print("Entrenando el algoritmo SVM")
    clf = svm.SVC(kernel='linear') 
    
    # Añadimos más caracteristicas al modelo
    #lstFeaturesTFIDFByComment = np.append(lstFeaturesTFIDFByComment, np.reshape(dataset["sarcasm"].to_numpy(dtype=int),(len(dataset),1)),axis=1)
    nCaracterist = ["argumentation","constructiveness","positive_stance",
                    "negative_stance","target_person","target_group","stereotype",	
                   "sarcasm", "mockery","insult","improper_language",
                    "aggressiveness","intolerance"]
    for car in nCaracterist:
        lstFeaturesTFIDFByComment = np.append(lstFeaturesTFIDFByComment, np.reshape(dataset[car].to_numpy(dtype=int),(len(dataset),1)),axis=1)
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy
    print("Calculando accuracy")
    #scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10,n_jobs=4)
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=2,n_jobs=4) 

    print("Mostrando metricas mas precisas y matriz de confusion")
    classNames = dataset.toxicity.unique() 
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 2,n_jobs=4)       
    clasificationReport = classification_report(lstObjetiveClass, classesPredicted, target_names=classNames)
    matrixConfusion = confusion_matrix(lstObjetiveClass, classesPredicted)

    print("RESULTADOS")
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    print("\nMostrando metricas mas precisas y matriz de confusion")
    print(clasificationReport)    
    print("Matriz de confusión con todo los datos")
    print(matrixConfusion)


# Modificación 4: modelo con todos los ajustes
def cuartoAjuste(path):
    #Leemos el archivo csv
    print("Leyendo detoxis dataset")
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')
    print(dataset.shape)

    #Procesamos los comentarios
    from nltk.stem.porter import PorterStemmer
    print("Procesando comentarios para extraer sus palabras")    
    stopWords = set(stopwords.words('spanish'))
    lstDocsDetoxis = []
    lstObjetiveClass = []
    for i in range(0, len(dataset)):
        review = limpiaAcentos(dataset['comment'][i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in stopWords]
        review = ' '.join(review)
        lstDocsDetoxis.append(review)
    
    # Crear la matriz de Words
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 10000) # limitacion de palabras utilizadas
    #cv = CountVectorizer(max_features = 5000) # limitacion de palabras utilizadas
    #cv = CountVectorizer(max_features = 1500) # limitacion de palabras utilizadas
    lstFeaturesTFIDFByComment = cv.fit_transform(lstDocsDetoxis).toarray()
    lstObjetiveClass = dataset["toxicity"].values


    # Dividir el data set en conjunto de entrenamiento y conjunto de testing
    from sklearn.model_selection import train_test_split
    lstFeaturesTFIDFByComment_train, lstFeaturesTFIDFByComment_test, lstObjetiveClass_train, lstObjetiveClass_test = train_test_split(lstFeaturesTFIDFByComment, lstObjetiveClass, test_size = 0.20, random_state = 0)
 

    #Configuramos el modelo de SVM para entrenar/predecir    
    from sklearn.naive_bayes import GaussianNB
    #print("Entrenando el algoritmo SVM")
    #clf = svm.SVC(kernel='linear') # Modelo de clasificación SVC
    print("Entrenando el algoritmo Native Bayes")
    clf = GaussianNB()     # Modelo de clasificación Bayesiana
    
    #Añadimos todas las caracterisiticas adionales que no sean categoricas
    nCaracterist = ["argumentation","constructiveness","positive_stance",
                    "negative_stance","target_person","target_group","stereotype",	
                    "sarcasm", "mockery","insult","improper_language",
                    "aggressiveness","intolerance"]
    for car in nCaracterist:
        lstFeaturesTFIDFByComment = np.append(lstFeaturesTFIDFByComment, np.reshape(dataset[car].to_numpy(dtype=int),(len(dataset),1)),axis=1)
    
    #Entrenar el modelo utilizando solo los datos de entrenamiento
    clf.fit(lstFeaturesTFIDFByComment_train,lstObjetiveClass_train)

    # Predicción de los resultados con el Conjunto de Testing
    lstObjetiveClass_pred  = clf.predict(lstFeaturesTFIDFByComment_test)

    # Elaborar una matriz de confusión con los datos de testing
    print("Calculando la clase objetivo con datos test")
    confusionMatrix = confusion_matrix(lstObjetiveClass_pred, lstObjetiveClass_test)    
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy con todos los datos
    print("Calculando Cross-Fold Validation")
    #scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10,n_jobs=4)
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=2,n_jobs=4) 
    
    print("calculando métricas más precisas")
    classNames = dataset.toxicity.unique() 
    #classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 10,n_jobs=4)       
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 2,n_jobs=4)       
    clasificationReport = classification_report(lstObjetiveClass, classesPredicted, target_names=classNames)
    matrixConfusion = confusion_matrix(lstObjetiveClass, classesPredicted)
    
    print("RESULTADOS")
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    print("\nMatriz de confusión utilizando los datos de prueba(independientes del modelo)")
    print(confusionMatrix)
    print("\nMostrando metricas mas precisas y matriz de confusion")
    print(clasificationReport)    
    print("Matriz de confusión con todo los datos")
    print(matrixConfusion)

# funcion para evaluar 10 cv para el modelo elegido
def modeloSeleccionado(path):
    #Leemos el archivo csv
    print("Leyendo detoxis dataset")
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')
    print(dataset.shape)

    #Procesamos los comentarios
    from nltk.stem.porter import PorterStemmer
    print("Procesando comentarios para extraer sus palabras")    
    stopWords = set(stopwords.words('spanish'))
    lstDocsDetoxis = []
    lstObjetiveClass = []
    for i in range(0, len(dataset)):
        review = limpiaAcentos(dataset['comment'][i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in stopWords]
        review = ' '.join(review)
        lstDocsDetoxis.append(review)
    
    # Crear la matriz de Words
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 5000) # limitacion de palabras utilizadas
    lstFeaturesTFIDFByComment = cv.fit_transform(lstDocsDetoxis).toarray()
    lstObjetiveClass = dataset["toxicity"].values


    # Dividir el data set en conjunto de entrenamiento y conjunto de testing
    from sklearn.model_selection import train_test_split
    lstFeaturesTFIDFByComment_train, lstFeaturesTFIDFByComment_test, lstObjetiveClass_train, lstObjetiveClass_test = train_test_split(lstFeaturesTFIDFByComment, lstObjetiveClass, test_size = 0.20, random_state = 0)
 

    #Configuramos el modelo de SVM para entrenar/predecir    
    from sklearn.naive_bayes import GaussianNB
    print("Entrenando el algoritmo SVM")
    clf = svm.SVC(kernel='linear') # Modelo de clasificación SVC
    
    #Añadimos todas las caracterisiticas adionales que no sean categoricas
    nCaracterist = ["argumentation","constructiveness","positive_stance",
                    "negative_stance","target_person","target_group","stereotype",	
                    "sarcasm", "mockery","insult","improper_language",
                    "aggressiveness","intolerance"]
    for car in nCaracterist:
        lstFeaturesTFIDFByComment = np.append(lstFeaturesTFIDFByComment, np.reshape(dataset[car].to_numpy(dtype=int),(len(dataset),1)),axis=1)
    
    #Entrenar el modelo utilizando solo los datos de entrenamiento
    clf.fit(lstFeaturesTFIDFByComment_train,lstObjetiveClass_train)

    # Predicción de los resultados con el Conjunto de Testing
    lstObjetiveClass_pred  = clf.predict(lstFeaturesTFIDFByComment_test)

    # Elaborar una matriz de confusión con los datos de testing
    print("Calculando la clase objetivo con datos test")
    confusionMatrix = confusion_matrix(lstObjetiveClass_pred, lstObjetiveClass_test)    
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy con todos los datos
    print("Calculando Cross-Fold Validation")
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10,n_jobs=4)
    #scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=2,n_jobs=4) 
    
    print("calculando métricas más precisas")
    classNames = dataset.toxicity.unique() 
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 10,n_jobs=4)       
    #classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 2,n_jobs=4)       
    clasificationReport = classification_report(lstObjetiveClass, classesPredicted, target_names=classNames)
    matrixConfusion = confusion_matrix(lstObjetiveClass, classesPredicted)
    
    print("RESULTADOS")
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    print("\nMatriz de confusión utilizando los datos de prueba(independientes del modelo)")
    print(confusionMatrix)
    print("\nMostrando metricas mas precisas y matriz de confusion")
    print(clasificationReport)    
    print("Matriz de confusión con todo los datos")
    print(matrixConfusion)

# funcion para predicir nuevos comentarios si son tóxicos o no
def getDisperseMatrix(path, addComponents):
    #Leemos el archivo csv
    dataset = pd.read_csv(path, dtype = str, encoding='utf-8')

    #Procesamos los comentarios
    from nltk.stem.porter import PorterStemmer
    print("Procesando comentarios para extraer sus palabras")    
    stopWords = set(stopwords.words('spanish'))
    lstDocsDetoxis = []
    for i in range(0, len(dataset)):
        review = limpiaAcentos(dataset['comment'][i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in stopWords]
        review = ' '.join(review)
        lstDocsDetoxis.append(review)
    
    # Crear la matriz de Words
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 5000) # limitacion de palabras utilizadas
    lstFeaturesTFIDFByComment = cv.fit_transform(lstDocsDetoxis).toarray() 
    if addComponents == True:
        #Añadimos todas las caracterisiticas adionales que no sean categoricas
        nCaracterist = ["argumentation","constructiveness","positive_stance",
                        "negative_stance","target_person","target_group","stereotype",	
                        "sarcasm", "mockery","insult","improper_language",
                        "aggressiveness","intolerance"]
        for car in nCaracterist:
            lstFeaturesTFIDFByComment = np.append(lstFeaturesTFIDFByComment, np.reshape(dataset[car].to_numpy(dtype=int),(len(dataset),1)),axis=1)
        
    return lstFeaturesTFIDFByComment


        

def run0(path_train,path_test):        
    lstFeaturesTFIDFByComment = getDisperseMatrix(path_train,False)
    dataset = pd.read_csv(path_train, dtype = str, encoding='utf-8') 
    lstObjetiveClass = dataset["toxicity"].values 
    
    disperseMatrix_test = getDisperseMatrix(path_test,False)
    
    #Configuramos el modelo de SVM para entrenar/predecir  
    clf = svm.SVC(kernel='linear') # Modelo de clasificación SVC
    #Entrenar el modelo utilizando solo los datos de entrenamiento
    clf.fit(lstFeaturesTFIDFByComment,lstObjetiveClass)
    
    # predecir nuevos datos
    classPredicted = clf.predict(disperseMatrix_test)

    
if __name__ == '__main__':
    
    # modelo básico
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #processDetoxisTraining(path)
    
    
   # primera modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #primerAjuste(path)
    
    # segunda modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #segundoAjuste(path)
    
    # tercera modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #tercerAjuste(path)

    # Cuarta modificación
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #cuartoAjuste(path)
    
    # Evaluacion de 10 folds Crsoss-Fold Validation
    #path = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    #modeloSeleccionado(path)
    
    # Predecir el corpus de testing
    path_train = "E:/SINE_Pract_2020_2021/dataset/train.csv"
    path_test = "E:/SINE_Pract_2020_2021/dataset/test.csv"
    run0(path_train,path_test)


