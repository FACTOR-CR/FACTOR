import numpy as np

### We wrote all our code based on what we learn last semester during
### the course "Vie Artificielle". We are trying to do a multiclass perceptron, 
### but our code is a bit slow.
### Please notify that we are only 4 in our group, we don't have a binome for 
### the preprocessing, so we tried our best the came up with the best results as possible
### in our situation

class model () :
    def __init__(self):
        '''
        This constructor initialize data members. 
        '''
        self.NBEPOCH = 100 # number of iterations
        self.NBCLASSES = 10 # number of classes
        self.is_trained=False
        self.w = np.array([]) # array of weights
   
    def fit(self,X,y) :
        '''
        This function train our model
        X: Training data matrix of dim num_train_samples * num_feat.
        y: Training label matrix of dim num_train_samples * num_labels.
        '''
       
        self.w = np.zeros((self.NBCLASSES,X.shape[1])) # initialization of wieghts
        
        arg_max, predicted_class = 0, 0 #intilization of argmax and predictions
        
        
        for it in range(self.NBEPOCH) :
            for j in range(X.shape[0]):
                # Prediction of the classe with a scalar (for each class)
                for c in range(self.NBCLASSES) :
                    cur = np.dot(X[j], self.w[c])
                    if cur >= arg_max :
                        arg_max, predicted_class = cur, c
                
                # If the prediction is not correct, adjust weights
                if (y[j] != predicted_class) :
                    self.w[int(y[j])] += X[j]*5
                    self.w[predicted_class] -= X[j]
    
    def predict(self,X):
        '''
        This function should provide predictions of labels on (test) data.
        '''
        
        y = np.zeros(X.shape[0])
        # For each data we predict the class and we put it in a array (with arg_max)
        for j in range(X.shape[0]) :
            arg_max, predicted_class = 0, 0
            for c in range(self.NBCLASSES):
                cur = np.dot(X[j], self.w[c])
                if cur >= arg_max :
                    arg_max, predicted_class = cur, c
            y[j] = predicted_class
        return y
        
        