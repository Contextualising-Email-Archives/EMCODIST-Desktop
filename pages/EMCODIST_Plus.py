import streamlit as st



import pandas as pd

import datetime
import logging
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
# import cosine similarity to check the proximity of sentences
from sklearn.metrics.pairwise import cosine_similarity
import spacy

Basic_path = ''
if 'shared' in  st.session_state:
    Basic_path = st.session_state.shared
    
INBOXES_DIR = Basic_path + '/OriginalEmails/enron_mail_20150507/maildir/'
DATASETS_DIR = Basic_path+'/ProcessedData/' # assuming all datasets are here
RESULTS_DIR = Basic_path+'/Results/Plus/'



# To use the following, first you need to import en_core_web_sm using 
# python -m Spacy download en_core_web_sm
# You may need to download separately
import en_core_web_sm
# logging.info(f"Load spacy addon: en_core_web_sm: {datetime.now()}")
spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS



# logging.info(f"START: Create SentenceTransformer: {datetime.now()}")
model = SentenceTransformer('bert-base-nli-mean-tokens')
# logging.info(f"END: Create SentenceTransformer: {datetime.now()}")

data_dict = {
    'news_Letters' : 'reminders_newletters.pkl',
    'energy_operations': 'energy_operations_research.pkl',
    'corporate_finance':  'corporate_finance.pkl',
    'energy_trade':  'energy_trade.pkl',
    'investments_commodities':  'investments_commodities.pkl',
    'top_management':  'top_management.pkl',
    'places_directions':  'places_directions.pkl',
    'leisure':  'leisure.pkl',
    'events_festivals':  'events_festivals.pkl',
    'reports_forecast_summaries':  'reports_forecast_summaries.pkl',
    'agreements_subpoena':  'aggrements_subpoena.pkl',
    'general':  'general.pkl',
    'inquiries_requests':  'inquiries_requests.pkl',
    'recruitment_performance_analysis':  'recruitment_performance_analysis.pkl',
    'payments_insolvency':  'payments_insolvency.pkl',
    'reports_drafts':  'reports_drafts.pkl',
    'forms_agreements':  'forms_aggrements.pkl',
    'reminders_newsletters':  'reminders_newletters.pkl',
}

RELEVANT_TOPICS = list(data_dict.keys())

#****************************************************************************#

# The following function is to get the email subset to serach the query.
# User supplies all the categories to search

def _get_records_df(df_list):

    if len(df_list) == 0:
        df =  pd.read_pickle(os.path.join(DATASETS_DIR, 'All_embedding_clusters_pkls_model2/', 'embeddings_full.pkl'))
        return df

    if 'all' in df_list:
        df =  pd.read_pickle(os.path.join(DATASETS_DIR, 'All_embedding_clusters_pkls_model2/', 'embeddings_full.pkl'))
        # logging.info('All: ', len(df)) #test - should print 241192
        return df
    else:
        df = pd.DataFrame()

        for name in df_list:
            if name in data_dict.keys():
                fileName = os.path.join(DATASETS_DIR, 'All_embedding_clusters_pkls_model2/', 'embeddings_'+data_dict[name])
                df = pd.concat([df,(pd.read_pickle(fileName))])
            else:
                logging.warn(f"Topic {name} not found in {data_dict.keys()}")

        
        if len(df_list) > 1:
            df = df.drop_duplicates(['public_id']).reset_index(drop=True)
            
        #else:  # default
        #    df =  pd.read_pickle(os.path.join(DATASETS_DIR, 'clusters', 'embeddings_full.pkl'))
        
    return df

#****************************************************************************#
#****************************************************************************#
# The following function is to convert date from string format to datetime format

def _get_date(x):
    default_date = 'Tue, 1 Jan 2222 00:00:00 -0000'
    try:
        result = pd.to_datetime(x, infer_datetime_format=True).date()
        
    except :
        result = pd.to_datetime(default_date, infer_datetime_format=True).date()
        
    return result


#****************************************************************************#

def make_clickable(link):
    
    # target _blank to open new window    
    ib = INBOXES_DIR.replace('/','\\')    
      
    # convert the url into link
    return '<a  target="_blank" href="{}"> Link </a>'.format(os.path.join(ib, link))

#****************************************************************************#
#The following function is to get emails similar to the query
def search_plus(query, df_list, from_date, to_date):
    #query => user given phrase
    #df_list => user selection
    #from_date =>user selection
    #to_date => user selection

    df = _get_records_df(df_list)
    from_date = _get_date(from_date)
    to_date = _get_date(to_date)
    # logging.info(f"\nfrom_date: {from_date}")
    # logging.info(f"\nto_date: {to_date}")
    df = df[(df['date'] >= from_date) & (df['date'] <= to_date)]
    df = df.reset_index(drop=True)
    # logging.info(df)
    
    len_df = len(df)
    logging.info(f"Length of df: {len_df}")

    if len_df == 0:
        logging.info("Returning an empty result set")
        empty_df = pd.DataFrame([], columns=[ 'relevance', 'email_id'])
        return empty_df.to_json(orient='records')

    content_embeddings = list(df['embeddings'])
    # logging.info(content_embeddings)
    qe = model.encode(query.lower())
    # logging.info(qe)
    c_s = cosine_similarity([qe, ], content_embeddings)

    results = [( c_s[0][i],i ) for i in range(1,len(c_s[0])) if c_s[0][i] >0.55 ] # 55% of the relevance is agreed
    results = sorted(results, reverse=True)

    new_df = pd.DataFrame(results, columns=[ 'relevance', 'email_id'])
    new_df['email_id'] = new_df['email_id'].apply(lambda x: df.iloc[x]['public_id']) # In the original file it is public_id 
    new_df = new_df[['email_id','relevance']]
    # Format relvance as a percentage for presentation purposes
    new_df['relevance'] = new_df['relevance'].apply(lambda x: round(x * 100))
    new_df = new_df.drop_duplicates(ignore_index=True)
    new_df =  new_df.reset_index(drop=True)
    
    # link is the column with hyperlinks
    new_df['Link'] = new_df['email_id'].apply(lambda x: make_clickable(x))
    new_df.to_csv(RESULTS_DIR+'Plus_'+query+'.csv')   
    
    
    return new_df



def main():

    st.set_page_config(layout="wide", page_title="Plus Page")
    st.write("# Welcome to EMCODIST Plus Model!")
    with  st.form(key="Form2"):
        c1, c2  = st.columns([3,1])

        with c1:
            query = st.text_input("Enter your search query 👇. For Best results limit query to two words", placeholder='Election Bush')

        with c2:
            dates = st.date_input ( 'Enter to and from search dates' , value=[datetime.date(2000,12,1), datetime.date(2004,10,9)] )

        c2, c3,c4,c5,c6 = st.columns(5)
        with c2:
            all = st.checkbox('all', value=True)
            top_management = st.checkbox('top-management')
        
        with c3:
            reports_drafts = st.checkbox('reports_drafts')
            reports_forecast_summaries = st.checkbox('reports_forecast_summaries')
            recruitment_performance_analysis = st.checkbox('recruitment_performance_analysis')

        with c4:
            energy_operations = st.checkbox('energy-operations')
            energy_trade = st.checkbox('energy-trade')
        
        with c5:
            corporate_finance = st.checkbox('corporate-finance')
            investments_commodities = st.checkbox('investments-commodities')
            payments_insolvency = st.checkbox('payments_insolvency')
        
        with c6:       
            news_Letters = st.checkbox('news-Letters')
            places_directions = st.checkbox('places_directions')
            leisure = st.checkbox('leisure')
            events_festivals = st.checkbox('events_festivals')
            general = st.checkbox('general')
            inquiries_requests = st.checkbox('inquiries_requests')
            reminders_newsletters = st.checkbox('reminders_newsletters')
        
    
        
        if dates:
            from_date = dates[0]            
            to_date = dates[1]
            

        check_list = [all,reminders_newsletters, inquiries_requests,general,events_festivals,leisure,top_management,
        places_directions, news_Letters, corporate_finance , investments_commodities, payments_insolvency,
        energy_operations, energy_trade, reports_drafts, reports_forecast_summaries , recruitment_performance_analysis
        ]

        search_list = []
        for key in check_list:
            if key:
                search_list.append([ i for i, j in locals().items() if j == key][0])
        
        query = query
        
        submit_button = st.form_submit_button(label = 'Search' )

    if submit_button:  
        new_df = search_plus(query=query, df_list = search_list, from_date = from_date, to_date = to_date )
        st.write('Your results file is at: '+RESULTS_DIR+'Plus_'+query+'.csv')

         
        new_df = new_df.to_html(escape=False)
        st.write(new_df, unsafe_allow_html=True)
        

        return

if __name__ == '__main__':
	main()
