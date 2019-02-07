#!usr/local/bin/python3
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os, sys, json, time, re, string
import json
import time
import itertools
import multiprocessing
import collections

import random
import nltk
#begin = time.time()
# NOTE : don't download every time
nltk.download('stopwords')
nltk.download('wordnet')

OUT_RESULTS = '/home/deshenghu/search-text/new-dataset/start_parse_and_cal/merge_part_final_score_text/final_check_page_score_output_52a5d'
os.makedirs(OUT_RESULTS, exist_ok=True)

def get_file_path_list(results_dir):
    fps = [os.path.join(results_dir, f) for f in os.listdir(results_dir)]
  
    print("this is the len of fps:", len(fps))
    return fps


def find_bad_text(s, txt_type):
    char_dic = collections.Counter(s)
    if txt_type=="snippet":
        if char_dic["{"]>2 or char_dic[":"]>10:
            return 1
    elif txt_type=="page":
        if char_dic["{"]>2 and char_dic[":"]>25:

            return 1
    elif txt_type=="meta":
        if char_dic["{"]>2 and char_dic[":"]>25:
            #print("bad ")
            return 1



# the calcalute the bias score of the text given the corresponding proportion threshold for unigram and bigram 
def cal_text_score(df_html, df_lexicon_ori, stops, wnl, tknzr, df_threshold, text, i, perc):
    # print("This is the text: ", text)
    qry = df_html["qry"].iloc[i]
    # print("This is the qry: ", qry)
    part_position = round(len(text)*perc*0.01)
    if part_position<2:
        text_score = 0
        return text_score
    else:
        text = text[:part_position]

    tokens = tknzr.tokenize(text)
    tokens = [wnl.lemmatize(token, "n") for token in tokens] # lemmatization noun
    tokens = [wnl.lemmatize(token, "v") for token in tokens] # lemmatization verb
    tokens = [token for token in tokens if token not in stops]

    # print("this is tokens: ", tokens)

    qry_t = tknzr.tokenize(qry)
    qry_t = [wnl.lemmatize(token, "n") for token in qry_t]
    qry_t = [wnl.lemmatize(token, "v") for token in qry_t]
    qry_t = [token for token in qry_t if token not in stops]

    qry_unigram_to_remove = qry_t

    df_lexicon_bigram = df_lexicon_ori[df_lexicon_ori["ngram"]==2]

    lexicon_bigram_list = df_lexicon_bigram['token'].tolist()

    qry_bigram_to_remove = [j for i in qry_t for j in lexicon_bigram_list if i in j]

    bigram_to_check_list = []

    for i in range(len(tokens)-1):
        bigram_to_check_list.append((tokens[i] + "_" + tokens[i+1]))
    # print("this is bigram_to_check_list:", bigram_to_check_list)


    df_bigram_token = pd.DataFrame(bigram_to_check_list, columns =['token'])


    df_bigram_overlap = pd.merge(df_bigram_token, df_lexicon_bigram, how ='inner', on = ['token'])

    bigram_overlap = df_bigram_overlap["token"].tolist()

    unigram_to_check_list = []


    if len(tokens)>1:
        for i in range(len(tokens)):
            if i==0:
                if (tokens[i]+"_"+tokens[i+1]) not in bigram_overlap:
                    unigram_to_check_list.append(tokens[i])
            elif i>0 and i<(len(tokens)-1):
                if (tokens[i-1]+"_"+tokens[i]) not in bigram_overlap and (tokens[i]+"_"+tokens[i+1]) not in bigram_overlap:
                    unigram_to_check_list.append(tokens[i])
            elif i==len(tokens)-1:
                if (tokens[i-1]+"_"+tokens[i]) not in bigram_overlap:
                    unigram_to_check_list.append(tokens[i])
    else:
        unigram_to_check_list= tokens

    # print("this is the unigram_to_check_list: ", unigram_to_check_list)


    df_unigram_token = pd.DataFrame(unigram_to_check_list, columns =['token'])

    df_unigram_overlap = pd.merge(df_unigram_token, df_threshold, how ='inner', on = ['token'])

    unigram_overlap = df_unigram_overlap["token"].tolist()

    df_bigram_overlap_new = pd.merge(df_bigram_token, df_threshold, how ='inner', on = ['token'])


    bigram_overlap_new = df_bigram_overlap_new["token"].tolist()

    # to remove the bigram and unigram that appear in the query 
    bigram_final = [j for j in bigram_overlap_new if j not in qry_bigram_to_remove]


    unigram_final = [j for j in unigram_overlap if j not in qry_unigram_to_remove]
    

    final_token_list = bigram_final + unigram_final

    #print("this is the final_token_list: ", final_token_list)

    df_final_token = pd.DataFrame(final_token_list, columns =['token'])

    df_merge = pd.merge(df_final_token, df_lexicon_ori, how ='inner', on = ['token'])

   
    # if there is no token appear in the lexicon, assign 0 to the text
    if df_merge['score'].count() < 1:

        text_score = 0
    else:
        text_score = df_merge['score'].sum()/df_merge['score'].count()

    #print("this is the text_score: ", str(text_score))
    return text_score

# check the type of the text and then deicde whether call the "cal_text_score" function or simply assign "np.nan" to it
def check_text_and_cal_score(df_html, df_lexicon_ori, stops, wnl, tknzr, threshold, text_flag, snip_flag, i, perc):

    if threshold==15:
        df_threshold = pd.read_csv("./votesmart_lexicon_top15.csv")
    elif threshold==10:
        df_threshold = pd.read_csv("./votesmart_lexicon_top10.csv")
    
    qry = df_html["qry"].iloc[i]

    if snip_flag==1:
        
        text = df_html["snippet"].iloc[i]
        if pd.isnull(text) or pd.isnull(qry) or pd.isnull(df_html['all_visible_text'].iloc[i]) or len(df_html['all_visible_text'].iloc[i])<300 or find_bad_text(text, "snippet")==1:
            text_score = np.NaN
        else:
            text = df_html["snippet"].iloc[i]
            
            text_score = cal_text_score(df_html, df_lexicon_ori, stops, wnl, tknzr, df_threshold, text, i, perc)



    if snip_flag ==0:
        #print("this is a page text")

        text_visible = df_html['all_visible_text'].iloc[i]
        if pd.isnull(text_visible) or len(text_visible)< 300 or pd.isnull(df_html['snippet'].iloc[i]) or pd.isnull(qry) or find_bad_text(text_visible, "page")==1: #find_bad_text(text, "page")==1
            text_score = np.NaN
        else:
           
            if text_flag==1:
                text = df_html['all_visible_text'].iloc[i]
                #print("this is all_visible_text")
            elif text_flag==2:
                text = df_html['all_meta'].iloc[i]
               

            text_score = cal_text_score(df_html, df_lexicon_ori, stops, wnl, tknzr, df_threshold, text, i, perc)


    return text_score 


def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)


def load_results(fps):
   
    out = itertools.chain.from_iterable([load_json(fp) for fp in fps])
    return pd.DataFrame(list(out))



def multi_worker(arg_tuple):
    fp_part, index_list = arg_tuple[0], arg_tuple[1]
    df_html_part = load_results(fp_part)


    lexicon_path = "./votesmart_lexicon_original.csv"

    
    # read lexicon
    df_lexicon_ori = pd.read_csv(lexicon_path)

    print("len(df_lexicon_ori): ", len(df_lexicon_ori))
    print("this is df_lexicon_ori[:2] : ", df_lexicon_ori[:10][["token", "ngram","score"]])


    stops = set(stopwords.words("english"))
    wnl = WordNetLemmatizer()
    tknzr = TweetTokenizer(preserve_case = False, reduce_len = True)

   
    per_list_1 = [1, 5, 10, 15, 20, 30,40]+[50, 60, 70, 80, 90, 100]

    score_thre_dic = dict()


    for i in [1,2]:
        for j in per_list_1:
            score_thre_dic["visible_text_score_thre"+str(i)+"_"+str(j)]=[]

        score_thre_dic["allmeta_score_thre"+str(i)+"_"+str(100)]=[]
        score_thre_dic["snip_score_thre"+str(i)+"_"+str(100)]=[]


    url_list=[]

    abstract_list=[]

    t1, t2 = 15, 10




    for i in range(len(df_html_part["url_x"])):
        if i%10==0:
            print("This is parsing %d item" % (i+index_list[0]))
        url_list.append(df_html_part["url_x"].iloc[i])
        abstract_list.append(df_html_part["snippet"].iloc[i])
       
        for k in per_list_1:


            visible_text_score_t1 = check_text_and_cal_score(df_html_part, df_lexicon_ori, stops, wnl, tknzr, t1, 1, 0, i, k)
            score_thre_dic["visible_text_score_thre1"+"_"+str(k)].append(visible_text_score_t1)
            

        allmeta_score_t1 = check_text_and_cal_score(df_html_part, df_lexicon_ori, stops, wnl, tknzr, t1, 2, 0, i, 100)
        score_thre_dic["allmeta_score_thre1"+"_"+str(100)].append(allmeta_score_t1)
          
        snip_score_t1 = check_text_and_cal_score(df_html_part, df_lexicon_ori, stops, wnl, tknzr, t1, True, 1, i, 100)
        score_thre_dic["snip_score_thre1"+"_"+str(100)].append(snip_score_t1)
           
        for k in per_list_1:


            visible_text_score_t2 = check_text_and_cal_score(df_html_part, df_lexicon_ori, stops, wnl, tknzr, t2, 1, 0, i, k)
            score_thre_dic["visible_text_score_thre2"+"_"+str(k)].append(visible_text_score_t2)
      

        allmeta_score_t2 = check_text_and_cal_score(df_html_part, df_lexicon_ori, stops, wnl, tknzr, t2, 2, 0, i, 100)
        score_thre_dic["allmeta_score_thre2"+"_"+str(100)].append(allmeta_score_t2)
      

        snip_score_t2 = check_text_and_cal_score(df_html_part, df_lexicon_ori, stops, wnl, tknzr, t2, True, 1, i, 100)
        score_thre_dic["snip_score_thre2"+"_"+str(100)].append(snip_score_t2)
           
    #print("this is the begining index %s and this is the visible_text_score_thre1_list %s:" % (str(index_list[0]), str(visible_text_score_thre1)))
    print("this is the url_list[0] %s :" % (str(url_list[0])))
    print("this is abstract_list[1] %s: " % (str(abstract_list[1])))

    for i in [1,2]:
        for k in per_list_1:
            df_html_part["visible_text_score_t"+str(i)+"_"+str(k)]= score_thre_dic["visible_text_score_thre"+str(i)+"_"+str(k)]
            
        df_html_part["allmeta_score_t"+str(i)+"_"+str(100)]= score_thre_dic["allmeta_score_thre"+str(i)+"_"+str(100)]
        df_html_part["snip_score_t"+str(i)+"_"+str(100)]= score_thre_dic["snip_score_thre"+str(i)+"_"+str(100)]



    print("calculating  score end!!")
    df_score = df_html_part.drop(columns=['all_visible_text', 'all_meta'])
    #df_score.rename(columns={'snippet':'abstract'}, inplace=True)

    gap_dict = dict()


    for i in [1,2]:
        for j in per_list_1:
            gap_dict["visible_text_score_t"+str(i)+"_"+str(j)]=[]
  
    for i in range(len(df_score)):

        for k in per_list_1:

            if pd.isnull(df_score['visible_text_score_t1'+"_"+str(k)].iloc[i]) or pd.isnull(df_score['snip_score_t1'+"_"+str(100)].iloc[i]):
                gap_dict["visible_text_score_t1"+"_"+str(k)].append(np.NaN)
            else:
                gap1 = df_score['visible_text_score_t1'+"_"+str(k)].iloc[i] - df_score['snip_score_t1'+"_"+str(100)].iloc[i]
                gap_dict["visible_text_score_t1"+"_"+str(k)].append(gap1)



        for k in per_list_1:

            if pd.isnull(df_score['visible_text_score_t2'+"_"+str(k)].iloc[i]) or pd.isnull(df_score['snip_score_t2'+"_"+str(100)].iloc[i]):
                gap_dict["visible_text_score_t2"+"_"+str(k)].append(np.NaN)
            else:
                gap2 = df_score['visible_text_score_t2'+"_"+str(k)].iloc[i] - df_score['snip_score_t2'+"_"+str(100)].iloc[i]
                gap_dict["visible_text_score_t2"+"_"+str(k)].append(gap2)

        

    for i in [1,2]:
        for k in per_list_1:
            df_score["gap_visible_t"+str(i)+"_"+str(k)]= gap_dict["visible_text_score_t"+str(i)+"_"+str(k)]




    file_name = os.path.join(OUT_RESULTS, "%s.csv" % str("part2_final_check_page_score_text_10_21_52a5d_update_"+str(index_list[0])+"_"+str(index_list[-1])))

    df_score.to_csv(file_name)

  

if __name__ == "__main__":
    
    # # get path for the latest lexicon and the json file which stores the extracted text from raw html 

    begin = time.time()
    JSON_DIR = "/home/deshenghu/search-text/new-dataset/start_parse_and_cal/parse_html_mp/10_21_data_parse_html_output_52a5d/json_dir"
    fps = get_file_path_list(JSON_DIR)

    N = len(fps)
    print('WIll Loaded %d EXTRACTED JSON FILES in total' % N)

     # Configure multiprocessing parameters
    numthreads = 30
    numlines = 5

    # Map multi_worker to data chunks
    pool = multiprocessing.Pool(processes=numthreads)
    index_list = list(range(N))
    # df.iloc[index_list]


    pool.map(multi_worker, 
        ((fps[line:line + numlines],index_list[line:line + numlines]) for line in range(0, N, numlines)))
    
    pool.close()
    pool.join()

    end = time.time()
    print("Total time is %f" % (end-begin)) 

