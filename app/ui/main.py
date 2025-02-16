import streamlit as st
import datetime
import pandas as pd
from snowflake.snowpark.context import get_active_session
from img_loader import render_image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import snowflake.permissions as permissions
import json
import time
import os

st.set_page_config(
    page_title='Data Mapping',
    layout="wide",
    initial_sidebar_state="expanded")
 
session = get_active_session()

def checkRefChanges():
    if 'REF_CHANGED' not in st.session_state:
        st.session_state['REF_CHANGED'] ='0'
    ref=permissions.get_reference_associations("AUDIENCE_DATA")
    changed=st.session_state['REF_CHANGED'] !=ref
    st.session_state['REF_CHANGED'] =ref
    return changed

def updateBrand(brand):
    session.sql(f"UPDATE OUT.SETTINGS SET BRAND = '{brand}' WHERE true ").collect()

def getBrand():
    cur=session.sql(f"SELECT BRAND FROM OUT.SETTINGS").collect()
    if len(cur)==0:
        session.sql(f"INSERT INTO OUT.SETTINGS VALUES ('Snowflake');").collect()
        cur=session.sql(f"SELECT BRAND FROM OUT.SETTINGS").collect()
    return cur[0][0]    

def init():
    firstLoading=False
    if 'FIRST_LOADING' not in st.session_state:
        st.session_state['FIRST_LOADING']=True
        firstLoading=True
    b=getBrand()
    filenames = os.listdir('./img/brands')
    filenames = [ file for file in filenames if file.endswith( ('.png','.jpg','.jpeg','.gif','.webp') ) ]
    brands=[os.path.splitext(filename)[0] for filename in filenames]
    brands= [b.title() for b in brands]
    brands.sort(key=lambda y: y.lower())
    filenames.sort(key=lambda y: y.lower())
    global ref
    ref=permissions.get_reference_associations("AUDIENCE_DATA")
    global CP_NAME    
    CP_NAME =  b        
    with st.sidebar: 
        placeholder = st.container()
        with st.expander("UI Settings"):
            if firstLoading==True:
                st.session_state['brs']=getBrand()
            brand=st.selectbox("Choose Branding",brands,key='brs')   
            updateBrand(brand)
            CP_NAME=  brand
        with placeholder:
            idx=brands.index(CP_NAME)
            imgb=filenames[idx]
            render_image(f'''brands/{imgb}''')  
        with st.expander("Setup Sample DB"):
            if not permissions.get_held_account_privileges(["IMPORTED PRIVILEGES ON SNOWFLAKE DB"]):
                    permissions.request_account_privileges(["IMPORTED PRIVILEGES ON SNOWFLAKE DB"])
            if not permissions.get_held_account_privileges(["CREATE DATABASE"]):
                    st.error("The app needs CREATE DB privilege to Create Sample DB")
                    permissions.request_account_privileges(["CREATE DATABASE"])
                    st.stop()
            if st.button('CREATE SAMPLE DB'):  
                res=session.sql("CALL out.sampledb()").collect()
                st.info('DATA_MAPPING_DB_TEST.PUBLIC.SALES_DATA successfully created!')
                if len(ref)==0:
                    permissions.request_reference("AUDIENCE_DATA")
                    # st.info("Please Select Existing Table containing Audience Data")
                    st.stop()
            if st.button('REVOKE FOR DELETION'):
                res=session.sql("CALL out.revoke_grants()").collect()
        if st.button("Change Data Source"):
            permissions.request_reference("AUDIENCE_DATA")
    st.markdown("# Data Preparation")  
    if len(ref)==0:
        st.info("Please Select Existing Table containing Audience Data")
        st.stop()

def get_automapping(colorig,coldest,model):
    with st.spinner(f"Asking LLM {mod} ... Please wait ..."):
        completion_model = f'{model}'
        prompt = f"""
            {datetime.datetime.now()} You are python developper, you will get 2 lists of columns name, do you best to match the 2 lists. Return a dict of the second list mapping with the first list. Only answer the dict nothing else. 
                                      Do not include column with no matching (None). Return the dict with double quote for key and value

        """
        promptall = f"""select snowflake.cortex.complete(
                                    '{completion_model}', 
                                    $$
                                        {prompt}
                                        ###
                                        first list is: {''.join(colorig)}
                                        second list is: {''.join(coldest)}
                                        ###
                                    $$) as ANSWER
                """

        return session.sql(promptall).collect()[0]["ANSWER"]

def checkViewGenerated(schema,view):
   return session.sql(f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.VIEWS 
        WHERE TABLE_SCHEMA = '{schema}' 
        AND TABLE_NAME = '{view}';
    """).collect()

def genView(ori,tgt):
    if len(ori) != len(tgt):
        raise ValueError("The number of original columns and new columns must match")
    column_mapping = ",\r\n\t".join(f"{orig} AS {new}" for orig, new in zip(ori, tgt))
    sql_stmt = f"""
    CREATE OR REPLACE VIEW OUT.{ '_'.join( CP_NAME.upper().split() )}_VIEW AS SELECT 
    \t{column_mapping} 
    FROM REFERENCE('AUDIENCE_DATA')
    """
    return sql_stmt    

def map_columns(colsO, colDest):

    # Vectorize descriptions using TF-IDF
    vectorizer = TfidfVectorizer().fit(colsO + colDest)
    tfidf_matrix_1 = vectorizer.transform(colsO)
    tfidf_matrix_2 = vectorizer.transform(colDest)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix_1, tfidf_matrix_2)

    # Create mapping based on highest similarity
    column_mapping = {}
    for i, row in enumerate(similarity_matrix):
        best_match_index = row.argmax()
        col_1 = colsO[i]
        col_2 = colDest[best_match_index]
        column_mapping[col_1] = col_2

    return column_mapping, similarity_matrix

def generate_mapping_function(column_mapping):
    mapping_lines = ["def map_columns(df_source, df_target):", "    # Column mapping"]
    mapping_lines.append(f"    column_mapping = {column_mapping}")
    mapping_lines.append("    for src_col, tgt_col in column_mapping.items():")
    mapping_lines.append("        if src_col in df_source.columns and tgt_col in df_target.columns:")
    mapping_lines.append("            df_target[tgt_col] = df_source[src_col]")
    mapping_lines.append("    return df_target")
    return '\n'.join(mapping_lines)

def get_fig_heatmap(similarity_matrix, columns_1, columns_2):
    fig = go.Figure(data=go.Heatmap(
                       z=similarity_matrix,
                       x=columns_2,
                       y=columns_1,
                       hoverongaps = False))
    fig.update_layout( margin=dict(l=20, r=60, t=0, b=20),showlegend=False,height=330)
    return fig

def get_view_btns(view_exist,r,el,pref,disabled=False):
    vName=f"{'_'.join( CP_NAME.upper().split() )}"
    with st.container():
        if view_exist:
            if el.button('Recreate View',use_container_width=True,key=pref+'_one',disabled=disabled):
                r=session.sql(r).collect()
                session.sql(f"""grant SELECT on view OUT.{vName}_VIEW to application role app_public;""").collect()
                res=session.sql(f"""SELECT * from OUT.{vName}_VIEW;""").collect()
                # el.header('Generated View:')
                # el.dataframe(res,use_container_width=True)
                st.experimental_rerun()
        if view_exist:        
            if el.button('Delete View',use_container_width=True,key=pref+'_two'):
                res=session.sql(f"""DROP VIEW OUT.{vName}_VIEW;""").collect()
                st.experimental_rerun()
        if not view_exist: 
            if el.button('Create View',use_container_width=True,key=pref+'_three',disabled=disabled):
                r=session.sql(r).collect()
                session.sql(f"""grant SELECT on view OUT.{vName}_VIEW to application role app_public;""").collect()
                el=session.sql(f"""SELECT * from OUT.{vName}_VIEW;""").collect()
                # st.header('Generated View:')
                # el.dataframe(res,use_container_width=True)
                st.experimental_rerun()

def get_description(pict,wid,desc):
    on=1;tw=3;th=43
    # on=st.slider('Columns settings',min_value=1,max_value=100)
    # tw=st.slider('Columns settings end',min_value=1,max_value=100)
    # th=st.slider('Columns settings end third',min_value=1,max_value=100)
    with st.container():
        col1,col2,col3=st.columns([on,tw,th])
        with col1:
            render_image('system/'+pict,wid) 
        with col3:
            st.markdown(desc)  
    st.divider()          

init()

cols_target=[       'Event timestamp, date, DATE, TS',
                    'User email address',
                    'Event item ID, item_id, ID_ITEM', 
                    'Event Item Quantity, QTY, Number',
                    'Store ID, could be Shop of plant',
                    'TRANS_ID, Transaction ID, Transac or Trans or Trans_ID',
                    'Event Item price, nominal price',
                    'Event Currency, curr,']

cols_target_label=[ 'EVENT_TIMESTAMP',
                    'USER_EMAIL',
                    'EVENT_ITEM_ID',
                    'EVENT_ITEM_QUANTITY',
                    'STORE_ID',                    
                    'TRANSACTION_ID',
                    'EVENT_ITEM_PRICE',
                    'EVENT_CURRENCY']

cols_origin = session.sql("SELECT * FROM REFERENCE('AUDIENCE_DATA') limit 1").to_pandas().columns.values.tolist()

vName=f"{'_'.join( CP_NAME.upper().split() )}"
view_exist=checkViewGenerated("OUT",f"{vName}_VIEW")[0][0]>0

tab1, tab2, tab3 = st.tabs(["Manual Mapping","Automatic Mapping - Cosine","Auto Mappping -  AI" ])

with tab3:
    models=['claude-3-5-sonnet','gemma-7b','snowflake-arctic','deepseek-r1']
    # mod=st.selectbox("Choose Model",models)
    mod=models[0]
    get_description('ai.png',100,f'''
    ## This Tab demonstrates an automatic mapping based on LLM.
    ### The {CP_NAME.title()} expected columns and Source table columns are passed as lists to Cortex LLM with a prompt instructing to find the mapping and return a Dict.
    ### You can click "CREATE VIEW" button to materialize the Mapping.
    ''')
    if 'AI_RES' not in st.session_state:
        st.session_state['AI_RES'] =get_automapping(cols_origin,cols_target_label,mod)
    if st.button("Refresh") or checkRefChanges()==True:
        st.session_state['AI_RES'] =get_automapping(cols_origin,cols_target_label,mod)   
    jres = json.loads(st.session_state['AI_RES'])
    cols_origin_AI = list(jres.keys())
    cols_target_label_AI = list(jres.values())   

    r=genView(cols_target_label_AI,cols_origin_AI)
    err=len(cols_target_label_AI)!=len(cols_target_label)
    if err:
        result= [item for item in cols_target_label if item not in cols_origin_AI]
        st.error("Missing mandatory fields: " + ' | \n'.join(result),icon="⚠️")

    st.subheader("Generated View Creation")
    st.code(r, language='sql')
    get_view_btns(view_exist,r,st,'AI',err)

with tab2:
    get_description('cosine.png',100,f'''
    ## This Tab demonstrates an automatic mapping based on Vector Similarity.
    ### The {CP_NAME.title()} expected columns and Source table columns are transformed in vectors, :blue[cosine_similarity] is used to propose a mapping. 
    ### You can click "CREATE VIEW" button to materialize the Mapping.
    ''')
    column_mapping, similarity_matrix = map_columns(cols_origin, cols_target)
    fig=get_fig_heatmap(similarity_matrix, cols_origin, cols_target_label)
    for key, value in column_mapping.items():
        column_mapping[key] = cols_target_label[cols_target.index(value)] 

    mapping_df = pd.DataFrame(list(column_mapping.items()), columns=['Column from Selected Table ', f'''Mapped to {CP_NAME.title()} Column'''])
    cols_origin_auto = mapping_df.iloc[:, 0].tolist() 
    cols_target_label_auto = mapping_df.iloc[:, 1].tolist()

    col1,col2,col3=st.columns(3)

    col1.subheader("Similarity Matrix Heatmap")
    col1.plotly_chart(fig, theme="streamlit",use_container_width=True)
    
    col2.subheader("Mapping Based on Name Similarity")
    col2.dataframe(mapping_df,use_container_width=True)
    r=genView(cols_origin_auto,cols_target_label_auto)
    col3.subheader("Generated View Creation")
    col3.code(r, language='sql')
    get_view_btns(view_exist,r,col3,'COS')

with tab1:
    get_description('manual.png',100,f'''
    ## This Tab demonstrates a manual mapping.
    ### For each Source table column you can select the corresponding column from the {CP_NAME.title()} expected shema columns.
    ### You can click "CREATE VIEW" button to materialize the Mapping.
    ''')
    st.subheader("Define Mapping (default selections are suggested by Automatic Mapping)")
    selected_values = {}
    num_columns=4
    columns = st.columns(num_columns)
    for i, col in  enumerate(cols_target_label):
        col_index = i % num_columns 
        with columns[col_index]:   
            try:
                idx=list(column_mapping.values()).index(col)
            except:
                idx=0   
            selected_values[col]=st.selectbox(f"Select mapping for {col}:", cols_origin,index=idx )
    cols_origin = list(selected_values.values())
    cols_target_label = list(selected_values.keys()) 
    reqManual=genView(cols_origin,cols_target_label)
    st.subheader("Generated View Creation")
    st.code(reqManual, language='sql')
    get_view_btns(view_exist,reqManual,st,'MANUAL')

if view_exist:
    curAppName=session.sql('SELECT CURRENT_DATABASE() ').collect()[0][0]
    snip=f'''
    CREATE OR REPLACE SHARE ACME_TO_{CP_NAME.upper()};
    CREATE OR REPLACE DATABASE DATA_MAPPING_SHARING;
    CREATE OR REPLACE SCHEMA DATA_MAPPING_SHARING.DATA;
    CREATE OR REPLACE TABLE DATA_MAPPING_SHARING.DATA.TO_BE_SHARED AS SELECT * FROM DATA_MAPPING_APP.OUT.{CP_NAME.upper()}_VIEW;
    GRANT USAGE ON DATABASE DATA_MAPPING_SHARING TO SHARE ACME_TO_{CP_NAME.upper()};
    GRANT USAGE ON SCHEMA DATA_MAPPING_SHARING.DATA TO SHARE ACME_TO_{CP_NAME.upper()};
    GRANT SELECT ON TABLE DATA_MAPPING_SHARING.DATA.TO_BE_SHARED TO SHARE ACME_TO_{CP_NAME.upper()};
    ALTER SHARE ACME_TO_{CP_NAME.upper()} ADD ACCOUNTS= <{CP_NAME.upper()}_ACCOUNT>;
    '''
    st.subheader(f'''Share Back to {CP_NAME.title()}''')
    st.code(snip, language='sql')



