import streamlit as st
import pandas as pd 
import matplotlib as plt
import pickle
import base64
import os
import json
import uuid
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, r2_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

header = st.beta_container()
dataset = st.beta_container()
model_load = st.beta_container()
model_scoring = st.beta_container()
download = st.beta_container()

# First define some helper functions to load in data and plot model metrics
@st.cache(persist = True)
def load_data(filename):
	data = pd.read_csv(filename)
	label = LabelEncoder()
		
	for col in data.columns:
			data[col] = label.fit_transform(data[col])

	return data

@st.cache(persist = True)
def load_data_semicolon(filename):
	data = pd.read_csv(filename, sep = ';')
	label = LabelEncoder()
		
	for col in data.columns:
			data[col] = label.fit_transform(data[col])

	return data

@st.cache(persist = True)
def pred_model(df):
	df_preds = pd.DataFrame(loaded_model.predict(df))
	df_preds.columns = ['class']
	df_appended = df.join(df_preds)

	return(df_appended)

import base64

import streamlit as st
import pandas as pd

@st.cache(persist = True)
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

with header:
    st.title("Shroom Classification Demo: Prediction Scoring App")
    st.markdown("This is a model scoring demo, for illustrative purposes only. This application allows you to utilize a machine learning model to predict whether mushrooms are edible or poisonous by uploading a csv file.")
    st.markdown("In order to generate predictions the model requires mushroom data in a specific format (i.e., the same format as the original data used to train the model: [Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)).")
    st.markdown("You'll then be able to upload a csv file in the same format. The application then prepares the data before calling the model to predict mushroom edibility. Finally, you can download the scored dataset.")
    st.markdown("**Disclaimer**: I don't even eat mushrooms and I've done about a whole of three minutes of research on the topic. It appears there's no actual golden rule to determine a shroom's edibility, so take from that what you will.")

with dataset:
    st.header("Input features")
    df = load_data('data/mushrooms.csv')
    class_names = ['edible', 'poisonous']
    y = df['class']
    X = df.drop(columns = ['class'])
    
    st.markdown("In order to succesfully add a prediction, the model requires the following features (columns) to be present:")
    st.table(X.columns)
    st.markdown("Furthermore, these columns should contain the following data, using the same notation:")

    st.markdown("**Attribute Information** (from [original description](https://archive.ics.uci.edu/ml/datasets/mushroom)):") 
    st.markdown("1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s")
    st.markdown("2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s")
    st.markdown("3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y")
    st.markdown("4. bruises?: bruises=t,no=f")
    st.markdown("5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s")
    st.markdown("6. gill-attachment: attached=a,descending=d,free=f,notched=n")
    st.markdown("7. gill-spacing: close=c,crowded=w,distant=d")
    st.markdown("9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y")    
    st.markdown("10. stalk-shape: enlarging=e,tapering=t") 
    st.markdown("11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?") 
    st.markdown("12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s") 
    st.markdown("13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s") 
    st.markdown("14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y") 
    st.markdown("15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y") 
    st.markdown("16. veil-type: partial=p,universal=u") 
    st.markdown("17. veil-color: brown=n,orange=o,white=w,yellow=y") 
    st.markdown("18. ring-number: none=n,one=o,two=t") 
    st.markdown("19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z") 
    st.markdown("20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y") 
    st.markdown("21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y") 
    st.markdown("22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d") 
   

with model_load:
    st.header("Model specifications")
    filename = 'model/shroom_classifier.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    st.markdown("Sci-kit learn's [gradient boosting machine algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) is used to generate predictions. This pre-trained model is being read-in rather than being trained in this application.")
    st.markdown("Before calling the model's predict function, uploaded data are first prepped by passing them through a label encoder. This simply converts the (textual) categorical data (described above) into numeric counterparts that the algorithm can 'read'.")

with model_scoring:
	st.header("Score the model: Generate predictions by uploading your csv file")
	st.markdown("For demo purposes: If you want an easy way to use this app you can simply download the original mushrooms data set (e.g., [from Kaggle](https://www.kaggle.com/uciml/mushroom-classification)) and manually slice off a part of the dataset (and remove the class label). Alternatively you can generate your own fake mushroom based on filling data for the listed features described above.")
	uploaded_file = st.file_uploader("Select csv file...", type="csv")
	if uploaded_file is not None:
		newdata = load_data_semicolon(uploaded_file)
		st.write("")
		df_predicted = pred_model(newdata)

with download:
    if uploaded_file is not None:
        st.markdown("**Classified shrooms:**")
        st.markdown("Data preview. See final column _class_ for edibility predictions")
        st.write(df_predicted.head(5))
        if st.button('Download data as CSV'):
            tmp_download_link = download_link(df_predicted, 'mushrooms_predicted.csv', 'Click here to download dataset')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

