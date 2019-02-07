#!usr/local/bin/python3
import os, json, sys
import time
import hashlib
import pandas as pd
import multiprocessing
from bs4 import BeautifulSoup
from datetime import datetime
import itertools

OUT_RESULTS = '/home/deshenghu/search-text/new-dataset/start_parse_and_cal/parse_html_mp/date_10_30_data_parse_html_output_872b/json_dir'
TXT_RESULTS = '/home/deshenghu/search-text/new-dataset/start_parse_and_cal/parse_html_mp/date_10_30_data_parse_html_output_872b/txt_dir'
os.makedirs(OUT_RESULTS, exist_ok=True)
os.makedirs(TXT_RESULTS, exist_ok=True)


# to extract all the visible text on a webpage 
def get_all_visible_text(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    all_str_list = []

    # to revmove the texts in some html tags that we don't want, e.g., text in the <script> tags  
    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title', "meta"])]
  
    for every_str in soup.stripped_strings:
        all_str_list.append(every_str)
    all_visible_text = " ".join(all_str_list)
    all_visible_text = " ".join(all_visible_text.split())
    #print(all_visible_text)
    
    return all_visible_text


# to extract the text of all the tag attributes, which are invisible to the page viewer 
def get_all_meta(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')

    head_tag =soup.find_all(True)[0]
    
    all_attrs_value_list = []


    for child in head_tag.descendants:
        # print("****************")
        # print("this is child's tag_name: ", child.name)
        if child.name is not None:
            #print("this is child's attrs_list: ", child.attrs)
            if child.attrs.keys() is not None and child.attrs:
               # print("this is the keys():", child.attrs.keys())
                merge_attr_value = []
                for each_key in child.attrs.keys():
                    if isinstance(child.attrs.get(each_key), str):
                        attr_value_str = child.attrs.get(each_key)
                    elif isinstance(child.attrs.get(each_key), list):
                        attr_value_str = " ".join(child.attrs.get(each_key))
                    merge_attr_value.append(attr_value_str)
              
                attris_value_str = " ".join(merge_attr_value)
                if len(attris_value_str)>10000:
                    attris_value_str = " "


                #print("this is the attris_value_str: ", attris_value_str)
                all_attrs_value_list.append(attris_value_str)
                
                #print("!!!!!!!!!!!!!end of attris_value_str!!!!!!!!!!!")
      
    
    all_meta = " ".join(all_attrs_value_list)
    
    return all_meta
    

# to extract the text of meta_description
def get_meta_description(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    
    description = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'}) or soup.find('meta', attrs={'name':'Description'}) or soup.find('meta', attrs={'name':'DESCRIPTION'})
    if description is not None:
        meta_description = description.get('content')
        if meta_description is not None:
            meta_description = str(meta_description) + '. '
        else:
            meta_description = "NULL"

    else:
        meta_description = "NULL"
        
    return meta_description




def load_lines(fp, file_type):
    with open(fp, 'r') as infile:
        if file_type=='json':
            return [json.loads(line) for line in infile]
        elif file_type=='text':
            return [line.strip() for line in infile]

def load_results(fps):
    
    out = itertools.chain.from_iterable([load_lines(fp, 'json') for fp in fps])
    return pd.DataFrame(list(out))

def load_results_dir(results_dir):

    fps = (os.path.join(results_dir, f) for f in os.listdir(results_dir))

    out = itertools.chain.from_iterable([load_lines(fp, 'json') for fp in fps])
    return pd.DataFrame(list(out))

def get_file_path_list(results_dir):
    fps = [os.path.join(results_dir, f) for f in os.listdir(results_dir)]
    
    print("this is the len of fps:", len(fps))
    return fps
   


def multi_worker(arg_tuple):
    fp_part, index_list = arg_tuple[0], arg_tuple[1]
   
    df_part = load_results(fp_part)

    df_merge_snippet= pd.merge(df_part, df_snippet, how ='inner', on = ['result_id'])

    df_merge_snippet.rename(columns={'serp_id_x':'serp_id'}, inplace=True)

    df = pd.merge(df_merge_snippet, df_qry, how ='inner', on = ['serp_id'])

    df_part = df

    exeption_txt_list, item_list = [], []

    begin = time.time()

    for i in range(len(df_part)):
        print("this is processing %d item of the json file %s:" % (i, (str(index_list[0])+"_"+str(index_list[-1]))))
        #print(df['url'][i])
        url = df_part['url'].iloc[i]
        url_x = df_part['url_x'].iloc[i]
        result_id = df_part['result_id'].iloc[i]
        snippet = df_part["text"].iloc[i]
        qry = df_part["qry"].iloc[i]
        serp_id_y = df_part["serp_id_y"].iloc[i]
        serp_id = df_part["serp_id"].iloc[i]
        crawl_id = df_part["crawl_id"].iloc[i]
        serp_rank_str = str(df_part["serp_rank"].iloc[i])
        cmpt_rank_str = str(df_part["cmpt_rank"].iloc[i])
        subrank_str = str(df_part["subrank"].iloc[i])
        title = df_part["title"].iloc[i]
        type_html = df_part["type"].iloc[i]
        directions_str = str(df_part["directions"].iloc[i])
        orient_str = str(df_part["orient"].iloc[i])

        try:
            all_visible_text = get_all_visible_text(df_part["html"].iloc[i])
        except:
            all_visible_text = "NULL"
            exeption_txt_list.append("hit a Exp when get_all_visible_text"+ str(df_part["url_x"].iloc[i])+"   "+result_id+"    end!")

        try:
            all_meta = get_all_meta(df_part["html"].iloc[i])
        except:
            all_meta = "NULL"
            exeption_txt_list.append("hit a Exp when get_all_meta"+ str(df_part["url_x"].iloc[i])+"   "+result_id+"    end!")

        try:
            meta_description = get_meta_description(df_part["html"].iloc[i])
        except:
            meta_description = "NULL"
            exeption_txt_list.append("hit a Exp when get_meta_description"+ str(df_part["url"].iloc[i])+"   "+result_id+"    end!")
            exeption_txt_list.append("hit a Exp when get_meta_description"+ str(df_part["url_x"].iloc[i])+"   "+result_id+"    end!")


        item = {'url': url, 'url_x': url_x, 'all_visible_text':all_visible_text, "all_meta":all_meta, 'meta_description':meta_description, 'result_id':result_id, "snippet":snippet, "qry":qry, "serp_id_y":serp_id_y, "serp_id":serp_id, "crawl_id":crawl_id, "serp_rank_str":serp_rank_str, "cmpt_rank_str":cmpt_rank_str, "subrank_str":subrank_str, "title":title, "type_html":type_html, "directions_str":directions_str, "orient_str":orient_str}
        
        item_list.append(item)

    

    print("this is len of item_list:", len(item_list))  

    json_file_name = os.path.join(OUT_RESULTS, "%s.json" % str("Date_10_30_872b_Extracted_text_"+str(index_list[0])+"_"+str(index_list[-1])))

    with open(json_file_name, 'w') as f:
        json.dump(item_list, f)

    end = time.time()

    txt_file_name = os.path.join(TXT_RESULTS, "%s.txt" % str("Date_10_30_872b_Extracted_text_exception_record_"+str(index_list[0])+"_"+str(index_list[-1])))

    with open(txt_file_name, "w") as f:
        f.write(' %f Seconds,'  % ((end - begin))) 
        f.write("\n\n")
       
        f.write(str(exeption_txt_list))


    print("finish this range: ", str(str(index_list[0])+"_"+str(index_list[-1])))
        
        




if __name__ == "__main__":

    begin = time.time()


    HTML_DIR = '/home/deshenghu/proj/search-audit/data/crawls/872b58680baa0189f588107982beb9fa4d204fb3a35ae2749a839805/results_html'

    fps = get_file_path_list(HTML_DIR)

    SNIPPET_DIR = '/home/deshenghu/proj/search-audit/data/crawls/872b58680baa0189f588107982beb9fa4d204fb3a35ae2749a839805/results'
    QRY_DIR = '/home/deshenghu/proj/search-audit/data/crawls/872b58680baa0189f588107982beb9fa4d204fb3a35ae2749a839805/serps'
    global df_snippet
    df_snippet = load_results_dir(SNIPPET_DIR)
    print("this is the len of df_snippet:", len(df_snippet))
    print("columns:", df_snippet.columns)

    global df_qry
    df_qry_temp = load_results_dir(QRY_DIR) 

    df_qry = df_qry_temp.drop(columns=['html'])


    N = len(fps)
    print('WIll Loaded %d HTML JSON FILES in total' % N)

    # Configure multiprocessing parameters
    numthreads = 30
    numlines = 5

    # Map multi_worker to data chunks
    pool = multiprocessing.Pool(processes=numthreads)
    index_list = list(range(N))



    pool.map(multi_worker, 
        ((fps[line:line + numlines],index_list[line:line + numlines]) for line in range(0, N, numlines)))
    
    pool.close()
    pool.join()

    end = time.time()
    print("Total time is %f" % (end-begin)) 


