import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import pickle


# html 태그 제거 함수
def dart_preprocess(text):
    result = re.sub(r"<.*?>", " ", text)
    result = result.replace("\n", "")
    result = re.sub(r"\s+", " ", result)
    return result

    # 테마명-연관어 dictionary 생성 함수


def theme_preprocessing(data):
    themes_lite = data.copy()
    themes_lite = themes_lite[["테마명", "연관어1", "연관어2", "연관어3", "연관어4", "연관어5"]]

    # '없음' 행 제거
    themes_lite = themes_lite[themes_lite["테마명"] != "없음"]

    # NaN 값 제거
    themes_lite = themes_lite.dropna()

    themes_lite = themes_lite.reset_index(drop=True)

    # 중복 연관어 목록 생성 (dictionary)
    keyword_unique = pd.melt(
        themes_lite,
        id_vars=["테마명"],
        value_vars=["연관어1", "연관어2", "연관어3", "연관어4", "연관어5"],
    )
    keyword_unique = keyword_unique.drop(axis=1, columns="variable")
    k = keyword_unique.groupby(["테마명"])["value"].value_counts()
    k = k[k > 2].to_frame().reset_index()
    co_related_keywords = k.groupby("테마명").value.apply(list).to_dict()

    return themes_lite, co_related_keywords

    # KR-SBERT 임베딩 벡터 유사도 계산 함수


def imbedding_sim(theme_key_doc, theme_key_dic, document, model):
    if theme_key_doc:
        theme_key_list = [
            " ".join([theme] + key) for theme, key in theme_key_dic.items()
        ]
    else:
        theme_key_list = [theme for theme in theme_key_dic.keys()]

    theme_embedding = model.encode(theme_key_list)
    corp_embedding = model.encode(document.report)

    # 문서 간 유사도 행렬
    num_themes = len(theme_key_list)
    num_documents = len(document.report)

    similarity_matrix = np.zeros((num_documents, num_themes))

    for i in tqdm(range(len(theme_key_list))):
        for j in range(len(document)):
            theme_vec = theme_embedding[i]
            corp_vec = corp_embedding[j]
            similarity_matrix[j, i] = dot(theme_vec, corp_vec) / (
                norm(theme_vec) * norm(corp_vec)
            )

    similarity_df = pd.DataFrame(
        similarity_matrix, index=document.corp_name, columns=theme_key_dic.keys()
    )
    return similarity_df

    # 유사도 최소 min_similarity 이상인 기업 상위 top_n개 추출 함수


def themed_stock(similarity_df, min_similarity, top_n):
    theme_corp_dict = {}
    for i in tqdm(similarity_df.columns):
        corp_list = list(
            similarity_df[i][similarity_df[i] > 0.4]
            .sort_values(ascending=False)
            .index[:20]
        )
        if corp_list:
            theme_corp_dict[i] = corp_list
    return theme_corp_dict

    # 네이버 테마 종목 정확도 계산 함수


def naver_accuracy(themed_stock_dic, naver_themed_stock):
    accuracy = {}
    common_themes = set(naver_themed_stock.keys()) & set(themed_stock_dic.keys())
    for theme in common_themes:
        total = len(themed_stock_dic[theme])
        if total == 0:
            continue
        cnt = 0
        for stock in naver_themed_stock[theme]:
            if stock in themed_stock_dic[theme]:
                cnt += 1
        result = cnt / total
        accuracy[theme] = round(result, 3)
    return accuracy


# 3주간의 테마 지수 계산 함수
def theme_index(theme_stock_dic, stock_price, start, end):
    # 종가 str -> float 변환
    stock_price[stock_price.columns[1:]] = (
        stock_price[stock_price.columns[1:]]
        .replace(",", "", regex=True)
        .astype("float")
    )
    # 6/19 ~ 6/23
    stock_price2 = stock_price[
        (stock_price["일자"] >= start) & (stock_price["일자"] <= end)
    ]
    # 날짜 컬럼을 인덱스로 설정
    stock_price2.set_index("일자", inplace=True)

    # 테마지수 : 테마에 속한 종목들의 주가 평균
    theme_index = pd.DataFrame(index=stock_price2.index)
    for theme, stock in theme_stock_dic.items():
        temp = stock_price2[stock_price2.columns.intersection(stock)]
        theme_index[theme] = list(temp.mean(axis=1))

    return theme_index
