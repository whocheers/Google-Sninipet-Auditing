import pandas as pd
import numpy as np
import requests
import os, sys, json, time
def get_ngram(s):
    return 2 if "_" in s else 1
# main
if __name__ == "__main__":

    # Get path.
    sys_path = sys.path[0]
    sep = sys_path.find("src")
    file_path = sys_path[0:sep]
    ## path for  the csv file storing the parsed tokens(i.e., unigram & bigram) in the votesmart corpus
    ngram_path = os.path.join(file_path, "data", "votesmart_token.csv")
    ## path for saving the csv file which will add the bias scores for the tokens 
    save_path = os.path.join(file_path, "data", "votesmart_token_bias_score.csv")
    # Read data.
    df = pd.read_csv(ngram_path)
    ## dem/rep column refer to the count of the token in the Democrat/Republican-authored text 
    df["base"] = df["dem"] + df["rep"]
  
    dem_sum = df["dem"].sum()
    rep_sum = df["rep"].sum()
    # probabilities of each token (i.e., unigram and bigram) appearing in Republican- and Democrat-authored text are denoted as rep_p and dem_p
    df["dem_p"] = df["dem"] / dem_sum
    df["rep_p"] = df["rep"] / rep_sum
    # the partisan bias score is calculated as (rep_p - dem_p)/(rep_p + dem_p), as described in the paper
    df["score"] = (df["rep_p"] - df["dem_p"]) / (df["rep_p"] + df["dem_p"])
    df = df.sort_values("base", ascending = False)
    print("number of tokens before filtering:", len(df))
    df = df[df["base"]>50]
    print("number of tokens after filtering:", len(df))
    df.to_csv(save_path)
