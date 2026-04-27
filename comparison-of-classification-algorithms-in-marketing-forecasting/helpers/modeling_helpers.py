import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from time import time

    

def timer(func):
    def wrap_fuc(*args,**kwargs):
        t1=time()
        result = func(*args,**kwargs)
        t2=time()
        return result,t2-t1
    return wrap_fuc

@timer
def model_training(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model

def save_training_time(model_str,time):
    with open("results/train_time.csv","a") as f:
        f.write(f"{model_str};{time}\n")
        
   
def save_model(model,file_name):
    
    with open(f"models/{file_name}.pkl",'wb') as f:
        pickle.dump(model,f)

def load_model(path_to_model):
    with open(path_to_model,'rb') as f:
        model = pickle.load(f)
        
    return model
