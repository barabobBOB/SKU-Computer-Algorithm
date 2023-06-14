import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_rows', 100) 
pd.set_option('display.max_columns', 100) 
pd.set_option('display.width', 1000) 

data=pd.read_csv('tmdb_5000_movies.csv',encoding='euc-kr')


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


count_vector=CountVectorizer(ngram_range=(1,3))
c_vector_genres=count_vector.fit_transform(data['장르'])

gerne_c_sim=cosine_similarity(c_vector_genres,c_vector_genres).argsort()[:,::-1]

def get_recommed_movie_list(df,movie_title,top=30):
    target_movie_index=df[df['제목']==movie_title].index.values
    sim_index=gerne_c_sim[target_movie_index,:top].reshape(-1) 
    sim_index=sim_index[sim_index!=target_movie_index] 
    result=df.iloc[sim_index].sort_values('추천점수',ascending=False)[:10]
    return result



print("마음에 들었던 영화를 조건에 맞게 입력하세요:")
movie=input()
temp=get_recommed_movie_list(data,movie_title=movie)
ans=[]
ans=temp.values.tolist()
ans=array(ans)



for i in range(10):
        if i==0:
            print('%50s %40s %35s %20s %14s %20s' % ('제목','장르','평점','평점투표 수','인기도','추천 점수'))
        else:
            print('%60s %50s %20s %20s %20.4s %20.4s' % (ans[i][1],ans[i][2],ans[i][3],ans[i][4],ans[i][5],ans[i][7]))
           