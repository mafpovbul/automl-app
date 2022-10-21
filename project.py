import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json
from Database import *
from Annotation import *
from Parser import *
from typing import List, Dict


##################################################################################
# Page layout
## Page expands to full width
st.set_page_config(page_title='CZ4031 Project 2')
sns.set(style="ticks")

##################################################################################
### GLOBAL VARIABLES DEF

### LOAD SAVED QUERIES
queries_path = "queries.json"
with open(queries_path, "r") as queries_file:
    QUERY_DICT = json.load(queries_file)
MAX_CONFIG_PARAMS = 7

PARAM_DICT = {"Bitmap Scan": "enable_bitmapscan",
               "Index Scan": "enable_indexscan",
               "Index-only Scan": "enable_indexonlyscan",
               "Sequential Scan": "enable_seqscan",
               "TID Scan": "enable_tidscan",
               "Hash Join": "enable_hashjoin",
               "Merge Join": "enable_mergejoin",
               "Nested-Loop Join": "enable_nestloop",
               "Hashed Aggregation": "enable_hashagg",
               "Materialization": "enable_material",
               "Explicit Sort": "enable_sort"
              }

POSTGRES_CONFIG_DICT = { "enable_bitmapscan": True,
                        "enable_hashagg": True,
                        "enable_hashjoin": True,
                        "enable_indexscan" : True,
                        "enable_indexonlyscan" : True,
                        "enable_material": True,
                        "enable_mergejoin": True,
                        "enable_nestloop": True,
                        "enable_seqscan": True,
                        "enable_sort": True,
                        "enable_tidscan": True
                         }



##################################################################################
### FUNCTION DEFINITIONS

def load_query(option: str) -> str:
    return QUERY_DICT[option]

def load_conf() -> Dict:
    config_path = "config.json"
    with open(config_path, "r") as conf_file:
        conf = json.load(conf_file)
    db_conf = conf["db"]
    return db_conf

def connect_db() -> Database:
    db_conf = load_conf()
    db = Database(db_conf)
    return db

def start(query: str, params: List[str]) -> None:

    # Create Postgres Connection
    db = connect_db()

    # Generate permutations for chosen parameters
    param_permutations = generate_params_permutations(params)

    # Get QEPs from DB for each permutation
    db_explanations = db.get_explanations(query, param_permutations)
    st.write(db_explanations)

    # Parse and annotate optimal QEP (index 0)
    # annotate_query_plan(db_explanations[0])

    # Parse and analyze AQPs (index 1:)
    # analyze_query_plans(db_explanations)


def annotate_query_plan(query_plan: str) -> None:
    parser = Parser()
    if query_plan != None:
        res = parser.annotate_plan(query_plan)
    with st.container():
        st.subheader("Query Plan Explanation")
        st.write(res)

def analyze_query_plans(query_plans: List[Dict])-> None:
    # Analyze cost of each AQP
    parser = Parser()
    query_costs = parser.generate_query_costs(query_plans)
    plot_query_plans(query_costs)
    return

def plot_query_plans(query_costs: List[float])-> None:
    # Plot costs of each AQP
    ### ADD CODE HERE ###

    return

def generate_params_permutations(params: List[str]) -> List[Dict]:
    # To generate optimal query plan:
    permutations = []
    permutations.append(POSTGRES_CONFIG_DICT.copy())

    # Generate other permutations of parameters for AQP
    # permutations.append( "other param permutations" )
    ### ADD CODE HERE ###

    return permutations

##################################################################################
st.write("""
# Query Comparison Annotator
This app provides a quick and easy way of retrieving the optimal Query Execution Plan (QEP) given a query,
and annotates the query based on why certain operators are chosen over Alternative Query Plans (AQP)
to make the execution more optimal.
""")

##################################################################################
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Select Database:'):
    data_type = st.sidebar.radio('Choose database', ["TPC-H"])

# Sidebar - Specify task
with st.sidebar.header('2. Load Query'):
    query_options = ["None"] + ["Query " + i for i in QUERY_DICT.keys()]
    query_option = st.sidebar.selectbox(
        'Which query would you like to load?',
        (query_options))


# Sidebar - Specify task
with st.sidebar.header('3. Configure Parameters'):
    num_params = st.sidebar.selectbox(
        'How many parameters would you like to configure?',
        ([str(i) for i in range(MAX_CONFIG_PARAMS)]))
    params = ""

    if num_params != '0':
        st.sidebar.write("Select at most " + num_params + " parameter/s")
        params = st.sidebar.multiselect("Choose parameters",
                                          (PARAM_DICT.keys()))


##################################################################################
# Main panel

# Displays the dataset
st.subheader('1. Enter your Query')
if query_option != "None":
    query = st.text_area(label="Input query here", value = load_query(query_option[-1]))
else:
    query = st.text_area(label="Input query here")

button = st.button("Submit", disabled=False)
if button:
    if len(params) <= int(num_params):
        start(query, params)
    else:
        st.warning("You can only select " + num_params + " parameters!", icon="⚠")




