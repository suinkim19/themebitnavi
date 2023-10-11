import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# 텍스트에서 특정 패턴을 만족하는 부분을 추출하는 함수
def pattern_match(text, pattern):
    match = re.search(pattern, text, re.DOTALL)
    extracted_text = ""
    if match:
        extracted_text = match.group(1).strip()
    else:
        print(f"Pattern not found.")

    return extracted_text


# GPT 출력에서부터 테마명, 연관어를 추출하는 함수
def theme_keyword_extract(themes):
    # 빈 리스트 생성
    theme_name = []
    related_words = []

    for theme in themes:
        # GPT 출력이 없는 경우 PASS
        if pd.isna(theme):
            theme_name.append(np.nan)
            related_words.append([np.nan] * 5)
            continue

        # 테마명이 '없음'인 경우 PASS
        if "없음" in theme:
            theme_name.append("없음")
            related_words.append([np.nan] * 5)
            continue

        else:
            try:
                # 주식 테마 키워드: 이후의 단어를 키워드로 추출
                theme_name.append(pattern_match(theme, r"주식 테마 키워드:\s*(.*?)\n"))

                # 연관어: 이후의 단어들을 연관어로 추출
                related_word = theme.split("연관어:")[1]
            except:
                continue

        # 쉼표 단위로 연관어 구분
        related_word = [text.strip() for text in related_word.split(",")]

        # 연관어가 5개 이상 추출된 경우 또는 5개 미만 추출된 경우에 대한 전처리
        if len(related_word) >= 5:
            related_word = related_word[:5]
        else:
            related_word = related_word + [""] * (5 - len(related_word))

        related_words.append(related_word)

    df = pd.DataFrame(related_words, columns=["연관어1", "연관어2", "연관어3", "연관어4", "연관어5"])
    df.insert(0, "테마명", theme_name)

    return df


# 코사인 유사도 계산 함수
def cosine_similarity_index(x, y, model):
    # 임베딩 모델로부터 입력 단어 또는 문장의 임베딩 계산
    vectors = model.encode([x, y])
    # 유사도 계산
    similarities = util.cos_sim(vectors, vectors)

    return similarities[0][1].item()


# Paired data에 대한 코사인 유사도 계산 함수
def max_cosine_similarity(pair_data, model):
    similarity = 0

    # 각 아이템들에 대해 코사인 유사도의 최댓값 계산
    for item1, item2 in pair_data:
        similarity += max(
            [cosine_similarity_index(item1, item2_x, model) for item2_x in item2]
        )

    return similarity / len(pair_data)
