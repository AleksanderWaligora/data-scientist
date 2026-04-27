import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def draw_subplots(df,features_2darray,plot_type,x,y):
    fig, axes = plt.subplots(x,y,figsize=(12,12))
    for i in range(len(features_2darray)):
        for j in range(len(features_2darray[i])):
            if features_2darray[i][j] is not None:
                col_name=features_2darray[i][j]
                plot_type(data=df,x=col_name,ax=axes[i][j])
                axes[i][j].set_title(col_name)
                axes[i][j].set_xticklabels(labels=axes[i][j].get_xticklabels(),rotation=60)
                
            else:
                fig.delaxes(axes[i][j])
    fig.tight_layout()


    
            

def feature_importance_plot(importance_list,features,model_name):
    feature_importance=[[feature,abs(importance)] 
                        for (feature,importance) in 
                        zip(list(features),importance_list)]
    feature_importance.sort(key = lambda x : x[1],reverse=True)
    
    df_fi = pd.DataFrame(feature_importance)
    fi, ax = plt.subplots(figsize=(10,10))
    fig = sns.barplot(df_fi,x=1,y=0,ax=ax)
    #fig.set_xticklabels(rotation=90,labels=df_fi[0])
    fig.set_ylabel(None)
    fig.set_xlabel(None)
    fig.set_title(f"{model_name} feature importance")
    plt.tight_layout()

def feature_importance_subplots(feature_importance_list,model_name_list,features):
    fig, axes = plt.subplots(len(feature_importance_list),1,figsize=(10,len(feature_importance_list)*10))
    for i,(feature_importance,model_name) in enumerate(zip(feature_importance_list,model_name_list)):
                feature_importance=[[feature,abs(importance)] for (feature,importance) in zip(list(features),feature_importance)]
                feature_importance.sort(key = lambda x : x[1],reverse=True)
                df_fi = pd.DataFrame(feature_importance)
                
                sns.barplot(df_fi,x=1,y=0,ax=axes[i])
                axes[i].set_title(model_name)
                axes[i].set_ylabel(None)
                axes[i].set_xlabel(None)