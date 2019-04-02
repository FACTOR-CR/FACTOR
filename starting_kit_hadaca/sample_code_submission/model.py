import numpy as np
import pickle
from os.path import isfile

### We wrote all our code based on what we learn last semester during
### the course "Vie Artificielle". We are trying to do a multiclass perceptron, 
### but our code is a bit slow.
### Update : Our code is faster now ! We changed a lot of thing in order to optimize our algorithm
### Please notify that we are only 4 in our group, we don't have a binome for 
### the preprocessing, so we tried our best the came up with the best results as possible
### in our situation

class model ():
    def __init__(self):
        '''
        This constructor initialize data members. 
        '''
        self.NBEPOCH = 100 # number of iterations
        self.NBCLASSES = 10 # number of classes
        self.is_trained=False # Used to know if the perceptron is trained or not
        self.w = np.array([]) # array of weights
        self.ETA = 1.7 # value of train rate
        
    def fit(self,X,y) :
        '''
        This function train our model
        X: Training data matrix of dim num_train_samples * num_feat.
        y: Training label matrix of dim num_train_samples * num_labels.
        '''
        self.w = np.zeros((self.NBCLASSES,X.shape[1])) # initialization of weights
        arg_max, predicted_class = 0, 0 #intilization of argmax and predictions
        nberreur = 0
        for it in range(self.NBEPOCH) :
            #print("Epoque n° :",it,"Nb d'erreur :",nberreur)
            nberreur = 0
            for j in range(X.shape[0]):
                # Prediction of the classe with a scalar (for each class)
                arg_max = 0
                for c in range(self.NBCLASSES) :
                    cur = np.dot(X[j], self.w[c])
                    if cur >= arg_max :
                        arg_max, predicted_class = cur, c
                
                # If the prediction is not correct, adjust weights
                if (y[j] != predicted_class) :
                    nberreur = nberreur + 1
                    self.w[int(y[j])] += X[j]*self.ETA
                    self.w[predicted_class] -= X[j]
        self.is_trained = True
                    
                    
    def predict(self,X) :
        '''
        This function should provide predictions of labels on (test) data.
        '''
        y = np.zeros(X.shape[0])
        # For each data we predict the class and we put it in a array (with arg_max)
        for j in range(X.shape[0]) :
            arg_max,predicted_class = 0,0
            for c in range(self.NBCLASSES):
                cur = np.dot(X[j], self.w[c])
                if cur >= arg_max :
                    arg_max, predicted_class = cur, c
            y[j] = predicted_class
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

    
## Here, we used a main function in order to determine the best value for ETA
## and best the number of iterations.
'''
from libscores import get_metric
metric_name, scoring_function = get_metric()
def main(D):
    
    model_dir = 'sample_code_submission/' 
    data_dir = '/home/ubuntu/Bureau/info232-projects/FACTOR/hadaca_input_data'
    data_name = 'hadaca'
    trained_model_name = model_dir + data_name
    # Détermination du learning-rate
    for p in np.arange(1.6,1.8,0.01) :
        M1 = model(p,100)
        X_train = D.data['X_train']
        Y_train = D.data['Y_train'].ravel()

        X_valid = D.data['X_valid']
        Y_valid = D.data['Y_valid'].ravel()

        X_test = D.data['X_test']
        Y_test = D.data['Y_test'].ravel()

        if not(M1.is_trained) or True:
            M1.fit(X_train, Y_train)


        Y_hat_train_facto = M1.predict(X_train) # Optional, not really needed to test on taining examples
        Y_hat_valid_facto = M1.predict(X_valid)
        Y_hat_test_facto = M1.predict(X_test)


        print("ETA = ",p, 'Training score for the train set =', scoring_function(Y_train, Y_hat_train_facto))'''