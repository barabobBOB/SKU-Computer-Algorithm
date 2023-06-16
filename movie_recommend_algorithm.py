import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def data_transform():
    pd.set_option('display.max_rows', 100) 
    pd.set_option('display.max_columns', 100) 
    pd.set_option('display.width', 1000) 

    data=pd.read_csv('tmdb_5000_movies.csv',encoding='utf-8')

    data=data[['영화번호','제목','장르','평점','평점투표 수','인기도','키워드']]

    m=data['평점투표 수'].quantile(0.9)
    data=data.loc[data['평점투표 수']>=m]
    C=data['평점'].mean()
    
    def weighted_rating(x,m=m,C=C):
        v=x['평점투표 수']
        R=x['평점']
        return (v/(v+m)*R)+(m/(m+v)*C)
    
    data['추천점수']=data.apply(weighted_rating,axis=1)


    data['장르']=data['장르'].apply(literal_eval) 
    data['키워드']=data['키워드'].apply(literal_eval)
    data['장르']=data['장르'].apply(lambda x : [d['name'] for d in x]).apply(lambda x: " ".join(x)) 
    data['키워드']=data['키워드'].apply(lambda x : [d['name'] for d in x]).apply(lambda x: " ".join(x))
    
    return data

def get_recommed_movie_list(movie_title,top=30):
    df = data_transform()
    count_vector=CountVectorizer(ngram_range=(1,3))
    c_vector_genres=count_vector.fit_transform(data_transform()['장르'])
    gerne_c_sim=cosine_similarity(c_vector_genres,c_vector_genres).argsort()[:,::-1]
    target_movie_index=df[df['제목']==movie_title].index.values
    sim_index=gerne_c_sim[target_movie_index,:top].reshape(-1) 
    sim_index=sim_index[sim_index!=target_movie_index] 
    result=df.iloc[sim_index].sort_values('추천점수',ascending=False)[:10]
    return result