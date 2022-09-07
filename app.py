# App created by Data Professor http://youtube.com/dataprofessor
# GitHub repo of this app https://github.com/dataprofessor/ml-auto-app
# Demo of this app https://share.streamlit.io/dataprofessor/ml-auto-app/main/app.py

import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

##################################################################################

# Page layout
## Page expands to full width
st.set_page_config(page_title='AutoML App')
sns.set(style="ticks")


##################################################################################

# Exploratory Analysis
def ea(df):
    #df = df.loc[:100]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('**1.2. Dataset dimension**')
    st.info(df.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    st.markdown('**1.4. Univariate Plots**:')
    var = st.selectbox(
        'Choose a variable',
        df.columns)
    with st.expander("See Plot", expanded=True):
        fig = plt.figure(figsize=(10, 4))
        sns.violinplot(x=var, data=df)
        st.pyplot(fig)


    st.markdown('**1.5. Bivariate Plots**:')
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox(
            'Choose X variable',
            df.columns)
    with col2:
        y_var = st.selectbox(
            'Choose Y variable',
            df.columns)
    with st.expander("See Plot", expanded=True):

        bi_fig = sns.jointplot(x=x_var, y=y_var, data=df, kind="reg")
        st.pyplot(bi_fig)

# Model building
def build_model(df, type):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]  # Using last column as target

    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)

    if type == "Regression":
        model = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    elif type == "Classification":
        model = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = model.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = model.fit(X_train, X_test, Y_train, Y_test)

    st.subheader('2. Table of Model Performance')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def load_example_data(type):
    if type == "Regression":
        df = load_diabetes()
        st.markdown('The Diabetes dataset is used as the example.')
    elif type == "Classification":
        df = load_iris()
        st.markdown('The Iris dataset is used as the example.')
    X = pd.DataFrame(df.data, columns=df.feature_names)
    Y = pd.Series(df.target, name='response')
    df = pd.concat([X, Y], axis=1)
    return df

##################################################################################
st.write("""
# AutoML App
This app provides a quick and easy way of initial exploration and modelling with automated feature engineering
using the **featuretools** library and fast comparisons of different machine learning algorithms using
the **lazypredict** library. Other functionalities like univariate and bivariate data visualisations 
are implemented as well. Developed by: Justyn Phoa
""")

##################################################################################
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Select Data type:'):
    data_type = st.sidebar.radio('', ["Example data", "Upload my data"])

uploaded_file = None
if data_type == "Upload my data" :
    with st.sidebar.header('1a. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

# Sidebar - Specify task
with st.sidebar.header('2. Select Task'):
    task_type = st.sidebar.radio('', ["Regression", "Classification"])

# Sidebar - Specify parameter settings
with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', value=80, min_value=0, max_value=100, step=1)
    seed_number = st.sidebar.number_input('Seed value', value=0, min_value=0, max_value=1000, step=1)

##################################################################################
# Main panel

# Displays the dataset
st.subheader('1. Dataset Exploratory Analysis')
if data_type == "Example data":
    df = load_example_data(task_type)
    st.write(df.head(5))
    ea(df)
    if st.button('Build models'):
        build_model(df, task_type)
elif data_type == "Upload my data":
    if uploaded_file is None:
        st.info('Awaiting for CSV file to be uploaded.')
    else:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        ea(df)
        if st.button('Build models'):
            build_model(df, task_type)



