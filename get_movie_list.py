from movie_recommend_algorithm import get_recommed_movie_list
from numpy import array

def movie_print():
    print("마음에 들었던 영화를 입력하세요:")
    movie=input()
    temp=get_recommed_movie_list(movie_title=movie)
    ans=[]
    ans=temp.values.tolist()
    ans=array(ans)

    for i in range(10):
            if i==0:
                print('%50s %40s %35s %20s %14s %20s' % ('제목','장르','평점','평점투표 수','인기도','추천 점수'))
            else:
                print('%60s %50s %20s %20s %20.4s %20.4s' % (ans[i][1],ans[i][2],ans[i][3],ans[i][4],ans[i][5],ans[i][7]))
           