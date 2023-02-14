# -*- coding: utf-8 -*-
"""
Spyder Editor
#Capstone Project Group 12
# -*- coding: utf-8 -*-
"""



"""
import sys, os, gc, traceback
from sklearn.utils import shuffle
from io import BytesIO
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from heartdisease  import PreProcessor, MissingValue, Outlier, visualize, encoder, FeatureEngineering, scaleing, modeler

os.chdir('C:\Users\Lenovo\Documents\Learning\Capstone Project\Heart Disease Prediction')


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Predicted_data', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Fraud_prediction.xlsx">Download csv file</a>'

def predict_batch_data(batch_data, modelName, response):
   
   pred_mapping = {0:'No CHD', 1:'CHD'}
   model = pickle.load(open(modelName, 'rb'))
   st.write('model loaded')

 # And Save policy number x_test firsts and then drop this column - For ml , it is not required

   
   feature_df  = batch_data.drop(columns=[response])
   target_df   = batch_data[response]
   
   
   st.write("Shape of the feature set" , feature_df.shape )
   st.write("No of classes" , len(np.unique(target_df)))
   #st.write("First 10 records of Apply data")
   #st.write(batch_data.head(10)) 
   y_pred = model.predict(feature_df)
   
   #batch_data['Predicted_Class'] = y_pred

   
   
   y_pred_list = [pred_mapping[i] for i in y_pred ]
   # predicted_df = feature_df
   batch_data['predicted_class'] = y_pred_list
   
   # Class distribution in apply data
   st.title('Class distribution after Prediction')
   #st.write("Class distribution in Predicted data")
   fig, ax = plt.subplots()
   st.write(sns.countplot(batch_data['predicted_class']))
   # Use Matplotlib to render seaborn
   st.pyplot(fig)
   
   # Predicting probabilities
   prob=model.predict_proba(feature_df)
   st.write("Model Probability" , prob )

   
   #return batch_data
   return batch_data


def streamlit_interface():
   """
      Function for Streamlit Interface
   """
   response ='TenYearCHD'
   
   st.markdown('<h1 style="background-color:lightblue; text-align:center; font-family:arial;color:white">CAPS Assignment - GROUP-99 </h1>', unsafe_allow_html=True)
   st.markdown('<h2 style="background-color:MediumSeaGreen; text-align:center; font-family:arial;color:white">Coronary Heart Disease Prediction</h2>', unsafe_allow_html=True)
   
   # Sidebars (Left)
   st.sidebar.header("Coronary Heart Disease")
   st.image('img1.png', width=600)

   # Sidebar -  Upload File for Batch Prediction
   st.sidebar.subheader("Get Batch Prediction")
   uploaded_file        = st.sidebar.file_uploader("Upload Your .csv File", type='csv', key=None)
   usr_sidebar_model    = st.sidebar.radio('Choose Your Model', ('Logistic Regression', 'LightGradientBoosting', 'Random Forest'))
   
   
   if uploaded_file is not None:
         batch_data = pd.read_csv(uploaded_file)
         
         batch_data = batch_data.dropna(0).reset_index(drop=True)
         numericalCols = batch_data.select_dtypes(include=["number"]).columns 
         dataNumeric = batch_data[numericalCols]
         
         st.write("Shape of the apply data set" , batch_data.shape )
         st.title("Apply data set rows")
         st.write(batch_data.head(15))
         
         st.title("Basic Stat")
         st.write(dataNumeric.describe())
         st.title("Unique Value in Each Column")
         st.write(batch_data.astype('object').nunique(axis=0))
         
    

         
   if st.sidebar.button('Submit Batch'):
      if uploaded_file is not None:
        #batch_data = pd.read_csv(uploaded_file)

         

        

        #st.write("Shape of the apply data set" , batch_data.shape )
        #st.title("Apply data set rows")
        #st.write(batch_data.head(15))
        # Perform batch Prediction
        batch_pred_df = predict_batch_data(batch_data, usr_sidebar_model + '.sav', response)

        # Save prediction
        # batch_pred_df.to_csv('./Fraud_prediction.csv', index=None)
        st.sidebar.text('Prediction Created Sucessfully!')
        
        st.header("Sample Output")
        st.write(batch_pred_df.head(10)) 
        
        #st.sidebar.text(shuffle(batch_pred_df.head()))
        st.sidebar.header("Download Complete File")
        #st.sidebar.markdown(get_table_download_link(batch_pred_df), unsafe_allow_html=True)
   
 
if __name__ == '__main__':
    streamlit_interface()