import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
영화 정보(이름, 장르, 점수 등) csv 파일을 전처리하는 함수입니다.
"""

def data_transform():
    # 행을 최대 100개까지 출력
    pd.set_option('display.max_rows', 100)
    # 열을 최대 100개 까지 출력
    pd.set_option('display.max_columns', 100)
    # 출력 창 넓이 설정
    pd.set_option('display.width', 1000) 

    # 영화 정보가 담긴 csv 파일(tmdb_5000_movies.csv)을 불러옵니다.
    # csv 파일의 출처는 다음과 같습니다.
    # https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv
    data = pd.read_csv('tmdb_5000_movies.csv',encoding='utf-8')

    # 사용할 열만 선택
    data = data[['영화번호','제목','장르','평점','평점투표 수','인기도','키워드']]

    '''
    투표 수가 많을수록 다양한 사람들의 의견이 반영되어 투표 결과가 편향되기 쉽습니다.
    그러한 불공정을 처리하기 위해 가중 평점(weighted rating) 방법을 이용합니다.
    
    R: 영화별 평점
    v: 영화별 평점을 투표한 횟수
    m: 순위 안에 들어야 하는 최소 투표 (500표)
    c: 전체 영화의 평균 평점
    투표 수가 상위 90프로 이상인 경우를 선정
    '''

    m = data['평점투표 수'].quantile(0.9)
    data = data.loc[data['평점투표 수']>=m]
    # 전체 영화의 평균 평점을 구한다.
    C = data['평점'].mean()
    
    # 가중 평점을 계산하는 함수
    def weighted_rating(x, m = m, C = C):
        v = x['평점투표 수']
        R = x['평점']
        return (v/(v+m) * R)+(m/(m+v) * C)
    
    # 가중 평점을 새로운 '추천점수' 열에 적용
    data['추천점수'] = data.apply(weighted_rating,axis=1)

    # '장르'와 '키워드' 열의 문자열을 파이썬 리스트 객체로 변환
    data['장르'] = data['장르'].apply(literal_eval) 
    data['키워드'] = data['키워드'].apply(literal_eval)
    
    # 각 '장르'와 '키워드' 항목에서 이름을 추출하여 새로운 문자열로 변환
    data['장르'] = data['장르'].apply(lambda x : [d['name'] for d in x]).apply(lambda x: " ".join(x)) 
    data['키워드'] = data['키워드'].apply(lambda x : [d['name'] for d in x]).apply(lambda x: " ".join(x))
    
    return data

"""
코사인 유사도 계산을 바탕으로 영화를 추천합니다.
"""
def get_recommed_movie_list(movie_title,top=30):
    # 데이터를 변환하는 함수를 호출하여 데이터프레임을 얻음
    df = data_transform()
    
    # CountVectorizer를 생성하여 ngram_range를 (1,3)으로 설정. 
    # 이는 개별 단어부터 3단어까지의 연속된 단어 그룹을 피처로 변환
    count_vector=CountVectorizer(ngram_range=(1,3))
    
    # 변환된 데이터의 '장르'열에 대해 CountVectorizer를 fit_transform을 수행. 장르에 대한 피처 벡터 행렬 생성
    c_vector_genres=count_vector.fit_transform(data_transform()['장르'])
    
    # 각 영화 간의 코사인 유사도 계산 후, 그 값을 내림차순으로 정렬
    gerne_c_sim=cosine_similarity(c_vector_genres,c_vector_genres).argsort()[:,::-1]
    
    # 특정 영화의 인덱스를 가져옴
    target_movie_index=df[df['제목']==movie_title].index.values
    
    # 가장 장르가 유사한 상위 'top' 개의 영화 인덱스를 가져옴. 이때, 자기 자신은 제외
    sim_index=gerne_c_sim[target_movie_index,:top].reshape(-1) 
    sim_index=sim_index[sim_index!=target_movie_index]
    
    # '추천점수'를 기준으로 상위 10개의 영화를 반환. 이 영화들이 추천 영화 리스트
    result=df.iloc[sim_index].sort_values('추천점수',ascending=False)[:10]
    
    return result