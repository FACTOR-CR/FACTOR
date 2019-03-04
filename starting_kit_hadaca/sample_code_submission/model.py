'''
Our classifier model : based on the multiclass perceptron
- fit: trains the model.
- predict : predict the class of datas
'''

import numpy as np
from sklearn.decomposition import TruncatedSVD

#Fonction utilitaires
def scalaire(x1, x2):
    '''
    Fonction qui retroune le produit scalaire 
    des deux vecteurs x1 et x2
    '''
    res = 0
    for i in range(x2.size) :
        res = res + (x1[i]*x2[i])
    return res 
def maxi (t) :
    '''
    Renvoie le maximun d'un tableau tab
    '''
    maxi = t[0]
    for i in range(t.size) :
        if t[i] > maxi :
            maxi = t[i]
    return maxi
def mini (t) :
    '''
    Renvoie le minimum d'un tableau tab
    '''
    mini = t[0]
    for i in range(t.size) :
        if t[i] < mini :
            mini = t[i]
    return mini

def probas(X, w) :
    '''
    Fonction qui permet de calculer la probabilité qu'une donnée data 
    appartienne à une classe, et qui renvoie le tableau de probabilités
    pour chaque classe.
    '''
    preds = np.zeros(10)
    scals = np.zeros(10)
    total = 0
    Max = 0
    A = 0
        
    for classe in range(10) :
        scals[classe] = scalaire(w[classe],X)
            
    Max = maxi(scals)
        
    for classe in range(10) :
        A = scals[classe] - Max
        total = total + np.exp(A)
            
    for classe in range(10) :
        A = scals[classe] - Max
        preds[classe] = np.exp(A) / total
    return preds
    
def maxIndice(preds) :
    '''
    Fonction qui renvoie l'indice de la case contenant 
    la valeur maximum d'un tableau preds
    '''
    indice = 0
    prediction = preds[0]
    for i in range(1,preds.size) :
        if (preds[i] > prediction) :
            prediction = preds[i]
            indice = i
    return indice
    
def updatew(X,y,w,preds) :
    '''
    Fonction qui permet de mettre à jour w, les poids, pour une donnée data et sa réference ref
    en fonction du tableau de prediction preds
    '''
    for classe in range(10) :
        for j in range(w.shape[1]) :
            if (y == classe + 1) :
                w[classe][j] += X[j]*(1-preds[classe])
            else :
                w[classe][j] -= X[j]*preds[classe]
    return w
     
def epoch(X,y,w) :
    '''
    Fonction epoch qui cherche pour chaque donnée la probabilité maximum et donc
    predit la classe de la donnée. Si la prédiction est fausse elle augmente le 
    nombre d'erreurs et appelle la fonction qui met a jour w.
    Elle renvoie le nombre d'erreur à chaque époque
    '''
    nbErr = 0
    x = 0
    preds = np.zeros(10)
    for i in range (X.shape[0]) :
        preds = probas(X[i], w)
        x = maxIndice(preds)
        if (x+1 != y[i]) :
            nbErr = nbErr + 1
            w = updatew(X[i], y[i], w, preds)
    return nbErr , w
    
    
#Classifier
class model () :
    def __init__(self):
        '''
        This constructor initialize data members. 
        '''
        self.NBEPOCH = 10
        self.NBCLASSES = 10
        self.is_trained=False
        self.svd = TruncatedSVD(n_components = 10)
        self.w = np.array([])

    
    def fit(self,X,y) :
        '''
        X: Training data matrix of dim num_train_samples * num_feat.
        y: Training label matrix of dim num_train_samples * num_labels.
        '''
        self.w = np.zeros((self.NBCLASSES,X.shape[1]))
        
        nbErr = 1
        iteration = 1
        while (iteration < self.NBEPOCH) :
            print(iteration)
            (nbErr, self.w) = epoch(X,y,self.w)
            iteration = iteration + 1

    
    def predict(self,X):
        '''
        This function should provide predictions of labels on (test) data.
        '''
        y = np.zeros(X.size)
        scal = np.zeros(self.w.shape[0])
        indice = 0
        for i in range(X.shape[0]) :
            for j in range(self.w.shape[0]) :
                scal[j] = scalaire(self.w[j],X[i])
            y[i] = maxIndice(scal)
        return y
 
        
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
