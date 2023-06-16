from movie_recommend_algorithm import get_recommed_movie_list
from numpy import array

"""
최근에 즐겼던 영화를 바탕으로 사용자에게 새로운 영화를 추천해주는 프로그램입니다.
"""
def movie_print():
    # 사용자로부터 영화 제목을 입력받음
    print("최근 마음에 들었던 영화를 입력하세요:")
    movie = input()
    
    # 입력받은 영화와 유사한 영화를 추천하는 함수를 호출함
    temp = get_recommed_movie_list(movie_title = movie)
    
    # 추천받은 영화 정보를 리스트로 변환
    ans = []
    ans = temp.values.tolist()
    ans = array(ans)

    # 상위 10개의 추천 영화 정보를 출력
    for i in range(10):
            if i == 0:
                # 표의 헤더를 출력
                print('%50s %40s %35s %20s %14s %20s' % ('제목','장르','평점','평점투표 수','인기도','추천 점수'))
            else:
                # 각 영화의 정보를 출력
                # ans[0]=영화번호, [1]=제목,[2]=장르,[3]=평점,[4]=평점투표 수,[5]=인기도,[6]=키워드,[7]=추천 점수
                print('%60s %50s %20s %20s %20.4s %20.4s' % (ans[i][1],ans[i][2],ans[i][3],ans[i][4],ans[i][5],ans[i][7]))
                

if __name__ == '__main__':
    movie_print()