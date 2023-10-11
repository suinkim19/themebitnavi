#######################################################
##               테마 네트워크 관련 함수              ##
#######################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


######################### 데이터 가져오기 및 전처리

def convert_dict_to_list(dictionary):
    """ 
    테마-연관어 딕셔너리를 문장으로 변환
    
    :param dictionary: 3회 이상 등장한 연관어에 대해 정리된 테마-연관어 딕셔너리
    :return: 테마명-연관어 딕셔너리를 문장들의 리스트
    """
    converted_list = []
    for key, values in dictionary.items():
        converted_list.append(key + ' ' + ' '.join(values))  # 테마명과 연관어를 띄어쓰기(' ')로 연결
    return converted_list  



######################### 테마지수 유사도 행렬 생성

def calculate_similarity(df):
    """ 
    테마지수 거리 계산 (유클리드 거리 기반)
    
    :param df: 최신 3주 간의 테마별 테마지수 값을 포함한 데이터프레임
    :return: 테마지수 유사도 행렬 (데이터프레임 형식)
    """

    # 테마명 가져오기
    themes = df.columns
    
    # 테마지수 유사도 행렬 생성
    similarity_matrix = pd.DataFrame(index=themes, columns=themes)
    
    # 테마별 테마지수 표준화
    scaler = StandardScaler()     # 표준화 객체 생성
    theme_standardized = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)       # 각 테마별로 표준화
    
    # 테마지수 거리 및 유사도 계산
    for theme1 in themes:
        for theme2 in themes:
            distance = np.linalg.norm(theme_standardized[theme1] - theme_standardized[theme2])   # 테마 간 유클리드 거리 계산
            similarity = 1 / (1 + distance)                      # 거리를 유사도로 변환 (1 / (1 + 거리))
            similarity_matrix.at[theme1, theme2] = similarity    # 테마지수 유사도 행렬에 해당 값 저장
    
    # 대각원소 0으로 만들기
    np.fill_diagonal(similarity_matrix.values, 0)
    
    return similarity_matrix


def plot_similarity_distribution(df, name):
    """ 
    유사도 행렬에서 유사도 값의 분포 시각적 확인
    
    :param df: 유사도 행렬(테이터프레임 형식)
    :param name: 시각화 결과 이미지 저장할 파일이름
    """
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 유사도 행렬의 상삼각 부분 추출 (df는 대칭행렬이므로 자료의 중복 방지하기 위해 상단 삼각형 부분의 자료만 사용)
    upper_triangle = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
    upper_triangle_values = upper_triangle.stack().values

    # 상단 삼각형의 값 분포를 히스토그램과 KDE(Kernel Density Estimation)로 시각화
    plt.figure(figsize=(5, 3))
    ax = sns.histplot(upper_triangle_values, bins=15, kde=False, color="#D6DCE5", edgecolor="white", stat='density')
    sns.kdeplot(upper_triangle_values, color="#344175", ax=ax)
    plt.title("")
    plt.xlabel('Similarity Values')
    plt.ylabel('Density')
    plt.savefig(name, bbox_inches='tight')




######################### 테마 네트워크 행렬 생성

def find_threshold_for_avg_degree(df, target_degree=5):
    """ 
    평균 연결(average degree)를 이용한 edge 필터링 기준값 찾기
        네트워크의 인접행렬(테마 네트워크 행렬)에서, 희소 네트워크로 만들기 위해 edge를 필터링합니다. 이를 위한 기준점을 찾는 함수입니다.
    
    :param df: 네트워크 행렬 데이터 (데이터프레임 형식). 유사도 값을 포함하며, 대칭 정방행렬이고, 대각원소가 모두 0이다.
    :param target_degree: 네트워크에서 목표로 하는 평균 연결 정도
    :return: 해당 테마 네트워크 행렬에 대한 적절한 임계값(필터링 기준)
    """
    
    df_filtered = df.copy()
    thresholds = np.arange(0.1, 1, 0.01)  # 임계치의 범위 (0.1부터 0.9까지 0.01씩 증가)
    
    for threshold in thresholds:
        df_filtered[df < threshold] = 0  # 임계치보다 작은 연결 제거
        G = nx.from_pandas_adjacency(df_filtered)  # 인접행렬을 네트워크로 변환
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()  # 평균 연결도 계산
        if abs(avg_degree - target_degree) < 1:
            return threshold

    # 적절한 임계치를 찾지 못한 경우
    return None


def filter_smiliarity(df, filtering = 0.3):
    """ 
    특정 임계값을 기준으로 네트워크의 약한 연결(작은 edge값) 제거
    
    :param df: 네트워크 행렬 데이터 (데이터프레임 형식).
    :param filtering: edge 값 제거할 기준값
    :return: 약한 연결이 제거된 희소 네트워크 행렬 (데이터프레임 형식).
    """
    
    # filtering 미만의 값 0으로 설정
    df_filtered = df.where(df >= filtering, 0)
    
    # 모든 값이 0인 테마 삭제
    ## 모든 값이 0이라는 의미는, 다른 테마와 연결이 전혀 없는 고립된 테마임을 의미합니다.
    ## 테마 네트워크 분석에서 이와 같이 다른 테마와 관련이 없는 고립된 테마는 관심 대상이 아니므로, 최종 결과에서 제거합니다.
    non_zero_rows_cols = (df_filtered != 0).any(axis=0)
    non_zero_rows_cols = non_zero_rows_cols[non_zero_rows_cols].index
    df_filtered = df_filtered.loc[non_zero_rows_cols, non_zero_rows_cols]
    
    return df_filtered


def min_max_scaling_matrix(df):
    """ 
    유사도 행렬의 값 0~1 범위로 표준화
    
    :param df: 유사도 행렬 데이터 (데이터프레임 형식).
    :return: 표준화된 유사도 행렬
    """
    
    # 유사도 행렬의 최소값과 최대값 찾기
    min_value = np.min(df)
    max_value = np.max(df)
    
    # 최소값과 최대값 이용하여 유사도 값을 0~1 범위로 표준화
    scaled_matrix = (df - min_value) / (max_value - min_value)
    
    return scaled_matrix


def top_similar_rows_with_values(df):
    """
    유사도 행렬에서 각 열별로 유사도가 높은 행과 그 값을 추출
    
    :param df: 유사도 행렬 (데이터 프레임 형식)
    :return: 각 열에 대한 유사한 행과 그 값을 포함하는 딕셔너리
    """
    result = {}
    for col in df.columns:
        # 0.0보다 큰 유사도 값을 가진 행만 정렬해서 추출
        sorted_rows = df[df[col] > 0.0][col].sort_values(ascending=False)
        result[col] = [(index, value) for index, value in sorted_rows.items()]
    return result


def generate_association_dataframe(output):
    """
    연관 행 정보를 담은 딕셔너리를 바탕으로 연관 테마 데이터프레임을 생성
    
    :param output: top_similar_rows_with_values 함수의 결과 (딕셔너리)
    :return: 연관 테마를 담은 데이터프레임
    """
    
    # 가장 많은 연관 테마를 가진 열의 길이를 파악
    max_len = max([len(val) for val in output.values()])
    
    # 연관 관계를 나타내는 데이터프레임 생성
    columns = ['연관테마' + str(i+1) for i in range(max_len)]
    df_association = pd.DataFrame(index=output.keys(), columns=columns)

    # 데이터프레임에 연관 테마 이름을 입력
    for key, values in output.items():
        for i, (row_name, _) in enumerate(values):
            df_association.loc[key, '연관테마' + str(i+1)] = row_name

    return df_association


def process_similarities(similarity):
    """
    유사도 행렬을 처리하여 연관 테마를 담은 데이터프레임을 반환
    
    :param similarity: 유사도 행렬
    :return: 연관 테마를 담은 데이터프레임
    """
    tmp = similarity.copy()
    tmp1 = top_similar_rows_with_values(tmp)    # 연관된 행을 찾고
    tmp2 = generate_association_dataframe(tmp1) # 데이터프레임을 생성
    return tmp2.fillna(" ")


def get_combined_theme_related_terms(dfs, theme_name):
    """
    주어진 여러 연관테마 데이터프레임 중에서 특정 테마와 관련된 테마들을 추출하여 새로운 데이터프레임을 반환
    
    :param dfs: 각 비율별 가중합 처리된 연관테마 데이터프레임 리스트
    :param theme_name: 테마명
    :return: 테마와 관련된 다른 테마를 포함하는 데이터프레임
    """
    
    # 결과를 저장할 데이터프레임 초기화
    max_columns = max([df.shape[1] for df in dfs])
    result_df = pd.DataFrame(index=[f"{i}:{10-i}" for i in range(11)], columns=[f"연관테마{i+1}" for i in range(max_columns)])
    result_df.index.name = "단어의미 : 테마지수"

    for idx, df in enumerate(dfs):
        if theme_name in df.index:
            terms = df.loc[theme_name].dropna().values
            result_df.iloc[idx, :len(terms)] = terms

    return result_df



######################### node, edge 각각의 속성 정보 포함한 데이터프레임 생성

## node 속성정보 관련 함수

def min_max_scaling(data, new_min=20, new_max=50):
    """ 
    시각화를 위해 node 크기를 재조정하는 함수
    
    :param data: 원본 데이터(노드의 크기)
    :param new_min: 재조정할 노드 크기의 최소값
    :param new_max: 재조정할 노드 크기의 최대값
    :return: 재조정된 count 값
    """
    old_min = min(data)
    old_max = max(data)
    
    scaled_data = []
    for x in data:
        if x <= 50:
            # 데이터 값을 new_min과 new_max 사이로 조정
            scaled_value = new_min + ((x - old_min) / (old_max - old_min)) * (new_max - new_min)
            scaled_data.append(scaled_value)
        else:
            # 데이터 값이 50을 초과하면 고정값 60으로 설정
            scaled_data.append(60)
    
    return scaled_data



# edge 속성정보 관련 함수

def get_pairs(df):
    """ 
    유사도 행렬에서 에지 정보를 추출하는 함수
    
    :param df: 유사도 행렬 (데이터프레임 형식)
    :return: (node1, node2, weight) 형태의 튜플 리스트
    """
    rows = []
    for i, row in df.iterrows():
        for j, value in enumerate(row):
            if value > 0:  # 유사도 값이 0보다 큰 경우에만 행 정보를 저장
                rows.append((i, df.columns[j], value))
    return rows

def process_edges(df):
    """ 
    에지 데이터를 처리하여 중복 및 불필요한 정보를 제거하고 정렬하는 함수
    
    :param df: 유사도 행렬 (데이터프레임 형식)
    :return: 정리된 에지 정보 데이터프레임
    """
    rows = get_pairs(df)    # 유사도 행렬에서 에지 정보 추출
    edges = pd.DataFrame(rows, columns=['node1', 'node2', 'weight'])
    
    # 중복 에지 정보 제거
    edges[['min_node', 'max_node']] = edges[['node1', 'node2']].apply(sorted, axis=1, result_type='expand')   # edges 가나다순 나열
    edges = edges.drop(['node1', 'node2'], axis=1)
    edges = edges.rename(columns={'min_node': 'node1', 'max_node': 'node2'})
    edges = edges.drop_duplicates(subset=['node1', 'node2'])   # 중복 엣지 정보 제거
    edges = edges[['node1','node2','weight']]
    edges = edges.sort_values(by=['node1', 'node2']).reset_index(drop=True)     # 노드별로 정렬
    return edges


def rescale_weight(weight):
    """ 
    연결 강도에 따라 에지 두께를 조정하는 함수
    
    :param weight: 에지의 연결 강도
    :return: 조정된 에지 두께
    """
    if weight < 0.5:
        return 1
    elif weight <= 0.6:
        return 5
    else:
        return 10
    
    

######################### 최종 결과 저장
    
def find_missing_themes(theme_counts_df, nodes_df, theme_index_date_df, count_filter=10):
    """ 
    count가 크지만 nodes_df에 누락된 테마를 찾는 함수
    
    :param theme_counts_df: 테마별 count 정보가 있는 데이터프레임
    :param nodes_df: 노드(node) 정보가 있는 데이터프레임
    :param theme_index_date_df: 테마지수 데이터프레임
    :param count_filter: 테마를 유효하다고 간주하는 최소 count
    :return: 누락되었지만 추가되어야 할 테마들의 목록
    """
    valid_themes = set(theme_counts_df[theme_counts_df['count'] >= count_filter]['테마명'])   # count가 10 이상인(유효한) 테마명 추출
    existing_nodes = set(nodes_df['node'])   # 현재 노드 데이터프레임에 존재하는 테마 목록 추출
    column_themes = set(list(theme_index_date_df.columns))  # theme_index_date의 컬럼명(테마명)을 set으로 변환
    missing_themes = valid_themes - existing_nodes  # 누락된 테마 찾기
    
    # column_themes 안에 포함된 테마명만 반환
        # count가 많다고 하더라도, 테마지수 데이터프레임(theme_index_date_df)에 포함되지 않은 것은 주식에 영향 미치지 않는 테마이므로, 해당 테마는 제거합니다.
    final_missing_themes = missing_themes.intersection(column_themes)
    
    return final_missing_themes


