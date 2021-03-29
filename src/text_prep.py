## text_prep.py
'''
A collection of helper functions for nlp preprocessing
'''

# Imports
import pandas as pd
import numpy as np
import re

# for parallelizing dataframe work
from multiprocessing import Pool

# multiprocessing.cpu_count() # 2 for colabs
# num_partitions = 100
# num_cores = 4

def parallelize_dataframe(df, func, num_cores=2, num_partitions=100):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    return(df)

def uppercase_proportion_column(s):
    '''
    given a pandas Series:
        containing rows of strings
    returns: a series of floats representing
        the percentage of capital letters vs total alpha chars
        in provided strings
    '''
    import re # dependent on re

    uc_pattern = '[A-Z]'
    alpha_pattern = '[A-Za-z]'

    cap_count = s.str.findall(uc_pattern).str.len()
    # print(cap_count)

    alpha_char_count = s.str.findall(alpha_pattern).str.len()
    # print(alpha_char_count)

    uc_proportion = cap_count / alpha_char_count
    # print(uc_proportion)

    return uc_proportion





# Convert all interior quotes to single quotes

def convert_interior_quotes(s):
    '''
    Arguments:
        s = Series of strings
            Takes a series of strings as an argument
            converts all interior quotes in a string to single quotes
    Returns: 
        Series of strings with interior quotes
    '''
    quotes_pattern = '["]+'
    return s.str.replace(quotes_pattern, "'")

def strip_ip(s):
    '''
    Arguments:
        s = Series of strings
            Takes a series of strings as an argument
            removes any ip addresses
    Returns: 
        Series of strings without ip addresses
    '''
    ip_pat = '(?:[0-9]{1,3}\.){3}[0-9]{1,3}'
    return s.str.replace(ip_pat, "")

def strip_url(s):
    '''
    Arguments:
        s = Series of strings
            Takes a series of strings as an argument
            removes any ip addresses
    Returns: 
        Series of strings without url
    '''
    url_pat = 'https?:\/\/\S*'
    return s.str.replace(url_pat, "")

def strip_whitespace(s):
    '''
    Arguments:
        s = Series of strings
            Takes a series of strings as an argument
            removes extraneous whitespace
    Returns: 
        Series of strings without extraneous whitespace
    '''
    
    t = s.copy()
    # remove whitespace from edge
    t = t.str.strip()

    # reduce interior whitespace to single space
    t = t.str.replace('[\s]+', ' ')

    return t


def remove_all_punct(s):
    '''
    Arguments:
        s = Series of strings
            Takes a series of strings as an argument
            removes all punctuation
    Returns: 
        Series of strings with no punctuation
    '''
    not_alpha_pattern = '[^A-Za-z\s]'
    return s.str.replace(not_alpha_pattern, "")

def tidy_series(s):
    '''
    returns tidied series
    '''
    # copy series
    t = s.copy()

    # call individual functions
    t = convert_interior_quotes(t)
    t = strip_whitespace(t)
    t = strip_ip(t)
    t = strip_url(t)

    return t