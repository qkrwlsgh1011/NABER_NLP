# <pythonfile>.ipynb notebook
# 프로그램 작성시에는 <pythonfile>.py
# python3 <pythonfile>.py
# streamlit run <streamlitapp>.py
# pip install pandas
# conda install pandas

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt


def text():
    #Mark Down
    st.markdown('네이버 증권에서 제공하는 종목 토론실에서 인기 검색종목 30개 기업을 뽑았습니다.')
    st.markdown('뽑은 30개 기업들에 대한 종목 토론실을 크롤링하여 댓글들의 데이터를 뽑았습니다.')
    st.markdown("뽑은 데이터들로 토큰화 작업을 진행 후 감성분석 작업을 실시하였습니다.")
    st.markdown("- 긍정 : 1:grinning:")
    st.markdown("- 중립 : 0:zipper_mouth_face:")
    st.markdown("- 부정 : -1:angry:")
    st.markdown("이모티콘은 댓글 성향에 따른 감성상태를 나타냅니다.")
    
    
def dataframe1():
    df = pd.read_csv('name_code_0206.csv', dtype=str)
    st.dataframe(df) # Same as st.write(df)


def stock_date_input():
    df_1 = pd.read_csv('name_code_0206.csv', dtype=str)
    df_2 = pd.read_csv('naver_up_down_end.csv')
    df_2.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    da = st.date_input(
        "날짜를 선택하세요",
        datetime.date(2023,2,9), max_value=datetime.date(2023,2,10), min_value=datetime.date(2023,2,1))
    st.write('선택한 날짜는:', da)
    
    stock = st.selectbox(
        '기업을 선택하세요',
        (df_1["종목명"]))
    st.write('선택한 기업은:',stock)
    
    df_3 = df_2[(df_2["기업명"]==f'{stock}') & (df_2["날짜"]==f'{da}')]
    shape = df_3.shape[0]
    st.dataframe(df_3)
    st.write(f"총 {shape}개의 행이 출력되었습니다")

def dataframe2():
    df = pd.read_csv('top30stock.csv')
    df.drop(["Unnamed: 0"],axis=1, inplace=True)
    st.dataframe(df)

def dataframe2_add():
    key = "unique_key"
    df_1 = pd.read_csv('name_code_0206.csv', dtype=str)
    stock = st.selectbox(
        '기업을 선택하세요',
        (df_1["종목명"]),key=key)
    
    df_2 = pd.read_csv('top30stock.csv')
    df_2.drop(["Unnamed: 0"],axis=1, inplace=True)
    check = st.multiselect(
        '원하시는 정보를 선택하세요',(df_2.columns[:]))
    st.dataframe(df_2[df_2["종목명"]==stock][check])
    df_3 = df_2[df_2["종목명"]==stock][check].shape[0]
    st.write(f"총 {df_3}개의 행이 출력되었습니다")
   
def makegraph():
    df_1 = pd.read_csv('name_code_0206.csv')
    df_2 = pd.read_csv('top30stock.csv')
    key = "key_1"
    st.subheader('기업별 종가그래프')
    option = st.selectbox(
        '기업을 선택하세요', (df_1["종목명"]),key=key)

    stock_data = df_2[(df_2['종목명']==option)][["날짜","종가"]]
    stock_data["날짜"] = pd.to_datetime(stock_data["날짜"])
    stock_data["종가"] = stock_data["종가"].str.replace(',',"")
    stock_data["종가"] = stock_data["종가"].astype(np.int64)
    x = range(len(stock_data["날짜"]))
    fig = plt.figure()
    
    plt.bar(x=x, height=stock_data["종가"], width=0.5, color='gray')
    plt.plot(x, stock_data["종가"], color='purple',marker="o")
    plt.xticks(x, stock_data["날짜"].dt.strftime("%Y-%m-%d"),rotation=90)
    plt.ylim(min(stock_data["종가"]-10000), max(stock_data["종가"]+3000))
    st.pyplot(fig)
   
  
def download(file):
    with open(file, 'r') as file:
        csv_data = file.read()
    st.download_button(
        label="Download File",
        data=csv_data,
        file_name='name_code_0206.csv',
        mime='text/csv',)
    
def token():
    df_1 = pd.read_csv("naver_up_down_end.csv").drop(["Unnamed: 0"],axis=1)
    df_1[["날짜","기업명","제목+내용","전날_대비_상승_하락"]]
    df_1.rename(columns={'전날_대비_상승_하락':'상승_하락'},inplace=True)
    st.markdown('------')
    st.header('댓글 정제화 작업')
    st.markdown('- 온전한 한글외에 모든 부분을 제거했다')
    st.markdown('- 초성,알파벳,특수문자,이모티콘 등을 제거했다')
    code = '''df_1['정제된 댓글'] = df_1['제목+내용'].str.replace('\\[삭제된 게시물의 답글\\]',' ')
                                    # 삭제된 게시물이 크롤링 된 경우가 있어 넣어주었다
                    df_1['정제된 댓글'] = df_1['정제된 댓글'].str.replace('제목+내용:',' ')
                    df_1['정제된 댓글'] = df_1['정제된 댓글'].str.replace('[^가-힣]',' ').str.replace(' +',' ').str.strip()
                    # 한글만 나오게하는 정규식을 사용했고 \r같은 특수문자들이 포함되어 나와 제거해주었다
                    df_1 = df_1[df_1['정제된 댓글'] != '']
                    # 내용이 없는 댓글을 제거해주었다
                    df_1 = df_1.reset_index(drop=True)
                    df_1[["날짜","기업명","정제된 댓글","상승_하락"]]'''
    st.code(code,language='python')
    st.markdown('작업 후')
    if st.button('정제화'):
        #df_1['정제된 댓글'] = df_1['제목+내용'].str.replace('\\[삭제된 게시물의 답글\\]',' ')
        #df_1['정제된 댓글'] = df_1['정제된 댓글'].str.replace('제목+내용:',' ')
        #df_1['정제된 댓글'] = df_1['정제된 댓글'].str.replace('[^가-힣]',' ').str.replace(' +',' ').str.strip()
        #df_1 = df_1[df_1['정제된 댓글'] != '']
        #df_1 = df_1.reset_index(drop=True)
        #df_1[["날짜","기업명","정제된 댓글","상승_하락"]]
        #st.write('실행 완료')
        filter = pd.read_csv("filter0220.csv").drop(["Unnamed: 0"],axis=1)
        filter
    else : 
        st.write('대기 중')
    
def token2(): 
    df_1 = pd.read_csv('name_code_0206.csv')
    key = 'shineekey'
    option = st.selectbox("기업을 선택하세요", (df_1["종목명"]),key=key)
    if st.button('토큰화'):
        df_2 = pd.read_csv("konlpydata.csv").drop(["Unnamed: 0.1"],axis=1)
        df_2.rename(columns={'전날_대비_상승_하락':'상승_하락'},inplace=True)
        df_2 = df_2[df_2["기업명"]==option]
        df_2[["날짜", "기업명","토큰화 댓글","상승_하락","label"]]
        st.write('실행 완료')

    else : 
        st.write('대기 중')

def konlpy():
    st.markdown('- 정제화를 완료했다면 konlpy를 이용해 댓글들을 토크나이징 해준다')
    code = '''def corpus_save(company):
    df = clean_sents_df(company) #댓글 정제한 함수입니다
    df['정제된 댓글 길이'] = [len(str(i)) for i in df['정제된 댓글']]
    
    # vocab.txt = 자연어 처리 사전입니다
    tp = [str(i) for i in list(df['정제된 댓글'])]
    save = '\ n'.join(tp)
    f = open("vocab.txt", 'a',encoding='utf8')
    f.write(save)
    f.close()
    
def corpus_init():
    f = open("vocab.txt", 'w',encoding='utf8')
    f.write('')
    f.close()
    for company in company_list:
        corpus_save(company)

def return_tokenizer():
    corpus = DoublespaceLineCorpus("vocab.txt",iter_sent=True)
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(corpus)
    scores = {word:score.score for word, score in nouns.items()}
    tokenizer = LTokenizer(scores=scores)
    return tokenizer'''
    st.code(code,language='python')
    
    st.markdown('----')
    
    st.markdown('''- 미리 준비한 긍정적 자연어 단어집과 부정적 단어집을 사용하여 
                댓글들에 대한 라벨링 마킹합니다''')
    code_1 = '''def labeling(target_df):
    f = open("/neg_pol_word.txt", 'r',encoding='utf8')
    words = f.readlines()
    f.close()
    neg_words_set = {word.strip('\n') for word in words}
    # 부정적 단어집 열기
    
    f = open("pos_pol_word.txt", 'r',encoding='utf8')
    words = f.readlines()
    f.close()
    pos_words_set = {word.strip('\n') for word in words}
    
    # 긍정적 단어집 열기
    label_score = []
    for token_list in target_df['토큰화 댓글']:
        sent_score = 0
        for token in token_list:
            if token in neg_words_set:
                sent_score -= 1
            elif token in pos_words_set:
                sent_score += 1
    # 댓글의 단어들을 단어집에서 찾아 부정적인 단어라면 -1 긍정적 단어라면 +1점을 매깁니다
   
        if sent_score < 0:
            label_score.append(-1)
        elif sent_score > 0:
            label_score.append(1)
        else:
            label_score.append(0)
    # 댓글의 단어들로 매긴 점수로 라벨링을 매깁니다
    
    target_df['label'] = label_score
    return target_df'''
    st.code(code_1,language='python')
#def image():
#    
#    image = Image.open('heechan.jpg')
#    st.image(image, caption='Sunrise by the mountains')


def main():
    st.title("자연어 처리/추천시스템 프로젝트:blue_book:")

    #st.sidebar.write('''
    ## lab1
    ## lab2
    #- lab3
    #- lab4
    #''')

    code = '''이번 프로젝트를 통해 네이버 종목토론실의 댓글을 통해 감성 분석을 사용해 \n다음날의 주가 상승률을 예측하고 크게는 추천해보는 프로젝트를 진행했습니다.'''
    st.code(code, language='python')
    st.markdown('------')
    
    download('name_code_0206.csv')
    st.markdown('30개 기업코드 다운로드')
    
    if st.checkbox("30개 기업확인"):
        dataframe1()
    
    text()
    st.markdown('------') 
    
    st.header('날짜/기업별 댓글 조회')
    st.markdown("네이버 종목토론실의 인기상위 30종목의 댓글을 수집했다")
    st.markdown("날짜/기업 조합으로 댓글 확인이 가능하다")
    stock_date_input()
    st.markdown('------')

    st.header('기업 주식정보')
    st.markdown("기업의 주식정보를 추가로 수집해 원하는 정보별로 확인가능하다")
    download('top30stock.csv')
    st.markdown('삼성 주식정보 다운로드')
    if st.checkbox("기업 주식정보"):
        dataframe2()
        
    dataframe2_add()   
    st.markdown('-----')
    makegraph()
    st.markdown('-----')

    st.header('전처리를 위한 댓글 확인')
    st.markdown('''댓글을 확인하면 초성,알파벳,특수문자 등등 정제가 필요한 정보들이 
                있음을 확인 할 수 있다''')    
    token()
    st.markdown('정제화가 된 댓글들을 확인 할 수 있다')
    st.markdown('------')
    st.header('토큰화 작업')
    konlpy()
    token2()
    st.markdown('토큰화 작업과 라벨링작업까지 완료')
    
    
if __name__ == "__main__":
    main()
