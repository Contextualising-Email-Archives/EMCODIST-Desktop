import streamlit as st
import pandas as pd
import datetime

from collections import defaultdict
import logging
import json
import os
import sys

import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd


#from display import frequency_plot, time_plot, create_wordcloud


Basic_path = ''
if 'shared' in  st.session_state:
    Basic_path = st.session_state.shared
    
INBOXES_DIR = Basic_path + '/OriginalEmails/enron_mail_20150507/maildir/'
DATASETS_DIR = Basic_path+'/ProcessedData/' # assuming all datasets are here
RESULTS_DIR = Basic_path+'/Results/Plus/'

#import en_core_web_sm
spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
nlp = spacy.blank('en')


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

    if len(df_list) == 0 or 'all' in df_list:
        df =  pd.read_pickle(os.path.join(DATASETS_DIR, 'Enron_deduplicated.pkl'))
        # logging.info('All: ', len(df)) #test - should print 241192
        return df
    else:
        df = pd.DataFrame()

        for name in df_list:
            if name in data_dict.keys():
                fileName = os.path.join(DATASETS_DIR, 'All_cluster_pkls', data_dict[name])
                df = pd.concat([df,(pd.read_pickle(fileName))])
            else:
                logging.warn(f"Topic {name} not found in {data_dict.keys()}")

        # if there are more than one topic is chosen
        if len(df_list) > 1:
            df = df.drop_duplicates(['public_id']).reset_index(drop=True)

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

#****************************************************************************#

def get_query_token_list(phrases_to_search):
    query_tokens_list = [nlp(item.lower()) for item in phrases_to_search]
    return query_tokens_list

#****************************************************************************#
# The following filter limits the user query upto three phrases and matches them in the text.
#The following function is especially useful when there are more phrases in the query
# Users are recommended to query with less number of phrases and use those important phrase at the start of the query
#text => query

def tokenize_filter(text, search_method):
    text = 'xxx' if text=='' else str(text)
    if len(text.split()) >2:
        text = [token for token in text.rstrip().lower().split() if token not in spacy_stopwords ]

    elif search_method=='full':
        text = [text.lower()]
    else:
        text = text.split()
    return text
#****************************************************************************#

#****************************************************************************#
# Following functions are to phrase match
# returns dictionary of queries and corresponding emails
# 

def find_a_match(row, matcher):
    text = row.content
    text = text[:8000] if len(text)>8000 else text
    doc = nlp(text)
    matches = matcher(doc)
    return bool(len(matches) > 0)

def get_emails_set(df, query, search_method):
    # print(f"query {query}")
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrases_to_search = tokenize_filter(query, search_method)
    # print(f"phrases_to_search {phrases_to_search}")
    # Create a list of tokens for each item in each query
    query_tokens_list = [nlp(item.lower()) for item in phrases_to_search]
    # print(f"query_tokens_list {query_tokens_list}")
    
    matcher.add("Queries", None, *query_tokens_list )
    

    emails_dict = defaultdict(set)

    for idx, row in df.iterrows():  # sequential search

        text =  row.content
        text = text[:8000] if len(text)>8000 else text
        doc = nlp(text)
        matches = matcher(doc)        
        
        # Create a set of the items found in the text
        found_items = set([doc[match[1]:match[2]] for match in matches])

        # Transform the item strings to lowercase to make it case insensitive
        for item in found_items:
            emails_dict[str(item)].add(row.public_id)  

        
    return emails_dict


# Now just save them as a excel, opening the list of row-ids.
#Actually the following procedure is not needed if we need only the list of public_ids.
#****************************************************************************#
def make_clickable(link):
    # target _blank to open new window
    ib = INBOXES_DIR.replace('/','\\')
    #print(ib)
    return f'<a target="_blank" href="{ib + link }"> Link </a>'


#****************************************************************************#

def search(query, search_list, from_date, to_date):
    #query ==>  query to be searched
    #search_list ==> list of topics options that users checked
    #from_date ==> from date to search
    #to_date ==> to date to search

    
    df = _get_records_df(search_list) # all records that the search to be performed
    from_date = _get_date(from_date)
    to_date = _get_date(to_date)
    df.sort_values(by=['date'])

    
    df = df[(df['date'] >= from_date) & (df['date'] <= to_date)]
    df = df.reset_index(drop=True)

    len_df = len(df)
    logging.info(f"Length of df: {len_df}")

    if len_df == 0:
        return json.dumps([])

    # TODO: Only need to support a single search term not multiples
    # by searching method = 'full' query first.
    
    

    # by searching method = 'full' query first.
    emails_dict = get_emails_set(df,query,'full') # sending for searching across
    if len(emails_dict.keys()) == 0 and len(query.split())>1: #search doesnt return any result
        emails_dict = get_emails_set(df,query,'split')

    new_df = pd.DataFrame([(key,pubid) for key,val in emails_dict.items() for pubid in list(val) ], columns = ('Query','public_id'))
    new_df = new_df[['public_id']].copy()
    
    new_df = new_df.drop_duplicates(ignore_index=True)
    
    new_df['Link'] = new_df['public_id'].apply(make_clickable)
    new_df =  new_df.reset_index(drop=True)
    try:
        new_df.to_csv(RESULTS_DIR+'Basic_'+query+'.csv')
    except:
        print('unable to write to Results folder. Try again after closing earlier results files')

    #

    return new_df




def main():

    st.set_page_config(layout="wide", page_title="Basic Page")
    #st.set_page_config(page_title="Basic Page", page_icon="🌍")

    st.write("# Welcome to EMCODIST Basic Model!")
    with  st.form(key="Form1"):
        c1, c2  = st.columns([3,1])

        with c1:
            query = st.text_input("Enter your search query 👇. For Best results limit query to two words", placeholder='Election Bush'  )
            
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
        
        #query = query
        #print('query 2= '+query)
        submit_button = st.form_submit_button(label = 'Search' )
        
        
    if submit_button:
        new_df = search(query=query, search_list = search_list, from_date = from_date, to_date = to_date )
        st.write('Your results file is at: '+RESULTS_DIR+'Basic_'+query+'.csv')

        new_df = new_df.to_html(escape=False)
        st.write(new_df, unsafe_allow_html=True)
        return 

        
if __name__ == '__main__':
	main()
    




