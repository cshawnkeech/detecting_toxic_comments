# imports
import spacy
from spacy.lang.en import English
from spacy.util import minibatch, compounding
from spacy.tokens import Doc
from spacy.scorer import Scorer

def evaluate(tokenizer, textcat, val_texts, val_cats, thresh=0.5):
    '''
    returns dictionary of evaluations for each category in textcat
    
    '''
    
    docs = (tokenizer(val_text) for val_text in val_texts)  
    
    # create dict of results
    evals_by_cat = dict()
    
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = val_cats[i]['cats']
        
        for label, score in doc.cats.items():
            
            # add label to dict if not already present
            if label not in evals_by_cat:
                evals_by_cat[label] = {'tp':0,
                                       'fp':0,
                                       'fn':0,
                                       'tn':0,}
            
            if score >= thresh and gold[label] >= thresh:
                evals_by_cat[label]['tp'] += 1

            elif score >= thresh and gold[label] < thresh:
                evals_by_cat[label]['fp'] += 1

            elif score < thresh and gold[label] < thresh:
                evals_by_cat[label]['tn'] += 1
            
            elif score < thresh and gold[label] >= thresh:
                evals_by_cat[label]['fn'] += 1
    
    for key in evals_by_cat.keys():

        tp = evals_by_cat[key]['tp']
        fp = evals_by_cat[key]['fp']
        fn = evals_by_cat[key]['fn']
        tn = evals_by_cat[key]['tn']
        
        # precision
        # edge case: avoid dividing by zero: precision = 1 when fp = 0
        if tp + fp == 0:
            evals_by_cat[key]['precision'] = 1
        else:    
            evals_by_cat[key]['precision'] = tp / (tp + fp)
        
        # recall
        # edge case: avoid dividing by zero: recall = 1 when fn = 0
        if tp + fn == 0:
            evals_by_cat[key]['recall'] = 1
        else:    
            evals_by_cat[key]['recall'] = tp / (tp + fn)
            
        precision = evals_by_cat[key]['precision']
        recall = evals_by_cat[key]['recall']
        
        if precision  + recall == 0:
            evals_by_cat[key]['f_score'] = 0.0
        else:
            evals_by_cat[key]['f_score'] = 2 * (precision * recall) / (precision + recall)
        
    evals_by_cat['TEXTCAT_LOSSES'] = losses['textcat']

    return evals_by_cat



def add_labels_helper(s):
    '''
    takes dataframe or series, 
    unpacks col labels and adds each as label to textcat
        formatted as uppercase
    '''
    
    for col in s.columns:
        print(col)
        textcat.add_label(col.upper())
        
        
def txt_and_multi_cat(txt_series, multi_cat_df):
    '''
    arguments: 
        txt_series: a pandas series with text to be categorized
        multi_cat_df: a series or df with categories and the results (typically one-hot, but booleans would likely work as well)
    
    returns: list of tuples in the format
        [(txt, {cats:{dict_of_cats:vals, cat2:val2}})]
    '''
        
    # convert each series or series slice to list
    t = txt_series.tolist()

    
    # get a category name for each dependent column
    cats = [multi_cat_df[cat].name.upper() for cat in multi_cat_df.columns]

    # create list of lists: validation values
    cat_vals = multi_cat_df.values.tolist()
    
    c = [{cats[i]: v for i, v in enumerate(row)} for row in cat_vals]
    c = [{'cats': i} for i in c]
    
    docs = list(zip(t, c))
    
    return docs