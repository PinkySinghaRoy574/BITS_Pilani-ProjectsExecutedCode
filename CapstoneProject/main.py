

import sys
import path
import pandas as pd
from fancyimpute import IterativeImputer
from fancyimpute import KNN 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 



def is_fileExists(filename):
    if path.isfile(filename):
        return True
    else:
        return False

    
def read_csv(filename):
    try:
        if (is_fileExists):
            fileHandler=pd.read_csv(filename)
            return fileHandler
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        sys.stderr.write("ERRORED DESC\t::%s:\n"% str(err))
        sys.stderr.write("ERRORED MODULE\t::%s:\n"%str(exc_type))
        sys.stderr.write("ERRORED LINE\t::%s:\n"%str(exc_tb.tb_lineno))
                            
def found_missing_series(fileHandler):
    try:
        dic_out = {}
        fileHandler.isna().sum()
        result=fileHandler.isna().sum()
        resultToDict=result.to_dict()
        for x, y in resultToDict.items():
            if y != 0:
                dic_out[x] = y
        result_series = pd.Series(dic_out) 
        return result_series
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        sys.stderr.write("ERRORED DESC\t::%s:\n"% str(err))
        sys.stderr.write("ERRORED MODULE\t::%s:\n"%str(exc_type))
        sys.stderr.write("ERRORED LINE\t::%s:\n"%str(exc_tb.tb_lineno))
                         
def get_columnNames(filePd):
    try:
        return (filePd.columns.tolist())
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        sys.stderr.write("ERRORED DESC\t::%s:\n"% str(err))
        sys.stderr.write("ERRORED MODULE\t::%s:\n"%str(exc_type))
        sys.stderr.write("ERRORED LINE\t::%s:\n"%str(exc_tb.tb_lineno))
    
def  get_dataType(filePd):
    print(filePd.dtypes)

def impute_data_mice(filePd,column_names):
    try:
        mice_imputer = IterativeImputer()
        arr = mice_imputer.fit_transform(filePd) 
        filePd = pd.DataFrame(arr, columns =column_names)
        return filePd
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        sys.stderr.write("ERRORED DESC\t::%s:\n"% str(err))
        sys.stderr.write("ERRORED MODULE\t::%s:\n"%str(exc_type))
        sys.stderr.write("ERRORED LINE\t::%s:\n"%str(exc_tb.tb_lineno))

def impute_data_knn(filePd,column_names):
    try:
        knn_imputer = KNN()
        arr = knn_imputer.fit_transform(filePd) 
        filePd = pd.DataFrame(arr, columns =column_names)
        return filePd
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        sys.stderr.write("ERRORED DESC\t::%s:\n"% str(err))
        sys.stderr.write("ERRORED MODULE\t::%s:\n"%str(exc_type))
        sys.stderr.write("ERRORED LINE\t::%s:\n"%str(exc_tb.tb_lineno))
                         
def draw_boxplot(col,title):
    sns.boxplot(col).set_title(title)
    plt.show()


def dbscan_outlier_pred(X, eps=0.5, min_samples=5): 
    data = X.dropna() #needs to change
    data = pd.DataFrame(scaler.fit_transform(data))
    pca = PCA(n_components=2)
    pca.fit(data)
    plt.scatter(pca.fit_transform(data)[:,0],pca.fit_transform(data)[:,1])
    df_pca = pd.DataFrame(pca.fit_transform(data)) 
    df_pca.columns = ['A1', 'A2']
    db = DBSCAN(eps=0.5, min_samples=5).fit(df_pca)
    labels = db.labels_
    colours = {} 
    colours[0] = 'r'
    colours[1] = 'g'
    colours[2] = 'b'
    colours[-1] = 'k'
    cvec = [colours[label] for label in labels] 
    r = plt.scatter(df_pca['A1'], df_pca['A2'], color ='r'); 
    g = plt.scatter(df_pca['A1'], df_pca['A2'], color ='g'); 
    b = plt.scatter(df_pca['A1'], df_pca['A2'], color ='b'); 
    k = plt.scatter(df_pca['A1'], df_pca['A2'], color ='k'); 
    # plt.figure(figsize =(15, 15)) 
    plt.scatter(df_pca['A1'], df_pca['A2'], c = cvec) 
    # Building the legend 
    plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1')) 
    plt.show()   
    
#main Method
file_name = "inputs/Heart Disease.csv"
pd_file=read_csv(file_name)
pd_file.head()
column_names = get_columnNames(pd_file)
print("\n Basic data ::\n{}".format(pd_file.describe()))
missing_values=found_missing_series(pd_file)
print(missing_values)
get_dataType(pd_file)
pd_file=impute_data_mice(pd_file,column_names)
missing_values=found_missing_series(pd_file)
print("Missing values after mice_imputer:::",missing_values)


for col_name in column_names:
     print("The column names are :::::::::::::::::::::",col_name)
     title = col_name+" outliers"
     draw_boxplot(pd_file[col_name], title)



