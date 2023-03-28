import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import time


def implement_model():
    for i in range(10):
        time.sleep(0.01)
        


def filterdata():
    data = pd.read_csv("mergescoredata.csv").drop(["Unnamed: 0"],axis=1)
    data = data.rename(columns={"긍정1 / 부정2 / 쓸데없는거3?_x": "긍정1,부정2"})
    counts = data["기업명"].value_counts()
    data = data[data["기업명"].isin(counts[counts > 100].index)]
    #data.to_csv("filter_data.csv")
    
def stock_data_input():
    data = pd.read_csv("filter_data.csv").drop(["Unnamed: 0"],axis=1)
    #data

    da = st.date_input(
        "날짜를 선택하세요",
        datetime.date(2023,2,1), max_value=datetime.date(2023,2,9), min_value=datetime.date(2023,2,1))
    # st.write('선택한 날짜는:', da)
    
    stock = st.selectbox(
        '기업을 선택하세요',
        (data["기업명"].unique()))
    # st.write('선택한 기업은:',stock)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text('모델에서 예측데이터 가져오는 중')

    for i in range(100):
        progress_bar.progress(i + 1)
        if (i + 1) % 10 == 0:
            status_text.text(f'로딩중 : {i + 1}%')
        implement_model()

    status_text.text('데이터 로딩 완료')
    
    result = data[(data["기업명"]==f'{stock}') & (data["날짜"]==f'{da}')]
    result.reset_index(inplace=True)
    
    if st.checkbox("내용 확인"):
        st.dataframe(result[["정제된 내용","score"]].head(10))
    
    conditions = [result["score"] > 55, result["score"] < 45]
    choices = ["positive", "negative"]
    result["conditions"] = np.select(conditions, choices, default="neutral")
        
    if result.empty:
        st.write("죄송합니다 댓글이 없습니다")
    else:
        labels = ['positive', 'negative', 'neutral']
        sizes = result["conditions"].value_counts()
        if "positive" not in sizes.index:
            sizes = sizes.append(pd.Series(0, index=["positive"]))

        if "neutral" not in sizes.index:
            sizes = sizes.append(pd.Series(0, index=["neutral"]))
        if "negative" not in sizes.index:
            sizes = sizes.append(pd.Series(0, index=["negative"])) 
        positive_size = sizes.loc["positive"]
        negative_size = sizes.loc["negative"]
        neutral_size = sizes.loc["neutral"] 
        
        fig, ax = plt.subplots()
        ax.pie([positive_size, negative_size, neutral_size], labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  
        #ax.text(-2, 0.5, f'positive: {happy}', ha='center', va='center')
        #ax.text(-2, 0.3, f'negative: {sad}', ha='center', va='center')
        #ax.text(-2, 0.1, f'neutral: {neutral}', ha='center', va='center')
        ax.legend(loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig)
        
        try:
            positive_percentage = round(result["conditions"].value_counts(10)[1], 3)
        except IndexError:
            positive_percentage = 0.0
        try:
            negative_percentage = round(result["conditions"].value_counts(10)[0], 3)
        except IndexError:
            negative_percentage = 0.0
        try:
            neutral_percentage = round(result["conditions"].value_counts(10)[2], 3)
        except IndexError:
            neutral_percentage = 0.0
        
        positive_md = f"{stock}의 긍정적인 반응은 <span style='color:blue'>{positive_percentage}</span> 입니다."
        negative_md = f"{stock}의 부정적인 반응은 <span style='color:orange'>{negative_percentage}</span> 입니다."
        neutral_md = f"{stock}의 중립적인 반응은 <span style='color:green'>{neutral_percentage}</span> 입니다."
        
        st.markdown(positive_md, unsafe_allow_html=True)
        st.markdown(negative_md, unsafe_allow_html=True)
        st.markdown(neutral_md, unsafe_allow_html=True)
        st.markdown('----')
    
    
    group = data[data["기업명"]==stock]
    group = group.groupby("날짜").mean()
    
    group["기업명"] = stock
    group['up_down'] = group['score'].apply(lambda x: 100 if x > 50 else -100)
    group.reset_index(inplace=True)
    group.loc[group["up_down"] < 0, "up_down"] = 0

    
    graph_1 = pd.read_csv('30stock.csv')
    zerone = pd.read_csv("01sai.csv")
    zerone.rename(columns={"new_col": "up_down"}, inplace=True)
    zerone = zerone[zerone["종목명"]==stock]
    zerone = zerone.reset_index()
    zerone.loc[zerone["up_down"] >= 0, "up_down"] = 100
    zerone.loc[zerone["up_down"] < 0, "up_down"] = 0
    zerone.rename(columns={"종목명": "기업명"}, inplace=True)
    zerone["날짜"] = pd.to_datetime(zerone["날짜"], format="%Y.%m.%d").dt.strftime("%Y-%m-%d")
    
    st.subheader('기업별 종가그래프')

    stock_data = graph_1[(graph_1['종목명']==stock)][["날짜","종가"]]
    stock_data["날짜"] = pd.to_datetime(stock_data["날짜"])
    stock_data["종가"] = stock_data["종가"].str.replace(',',"")
    stock_data["종가"] = stock_data["종가"].astype(np.int64)
    x = range(len(stock_data["날짜"]))
    fig = plt.figure()
    
    plt.bar(x=x, height=stock_data["종가"], width=0.5, color='gray')
    plt.xticks(x, stock_data["날짜"].dt.strftime("%Y-%m-%d"),rotation=90)
    plt.ylim(min(stock_data["종가"]-10000), max(stock_data["종가"]+3000))
    st.pyplot(fig)
    
    
    st.subheader("모델 예측 값의 날짜별 평균")
    group[["날짜","기업명","score","up_down"]]
    st.write("모델이 댓글로 예측한 점수를 50점을 기준으로 이상일시 +100 미만일때 0으로 표현하였다.")
    zerone[["날짜", "기업명", "종가","up_down"]]
    st.write("실제 종가의 전날 기준으로 상승시 +100 하락시 0으로 표현하였다.")
    
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(group["날짜"], group["up_down"], label="Group")
    ax.plot(zerone["날짜"], zerone["up_down"], label="Zerone")
    ax.legend(loc="best")
    plt.xticks(rotation=90)
    plt.ylim(-10, 150)
    # Display plot in Streamlit
    st.pyplot(fig)
    # Find intersection points
    #intersections = pd.merge(group, zerone, on="날짜")
    #x_intersections = intersections["날짜"].values
    #y_intersections = intersections["up_down_x"].values

    #fig, ax = plt.subplots()
    #ax.plot(group["날짜"], group["up_down"], label="Predict")
    #ax.plot(zerone["날짜"], zerone["up_down"], label="Real result")
    #    
    ##ax.scatter(x_intersections, y_intersections, color='red', marker='o')
    #ax.legend(loc='best')
    #plt.xticks(rotation=90)
    #plt.ylim(-120, 150)
    ##ax.set_xlabel("날짜")
    ##ax.set_ylabel("up_down")
    ##ax.set_title(f"{stock} 주식 변동 예측")

    ## Display plot in Streamlit
    #st.pyplot(fig)
    
    





st.title("모델 발표")
stock_data_input()


