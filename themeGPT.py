import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from clova_summary import clova_news_summary
import openai
import time

######################### themeGPT1 : 추출


def themeGPT1(data, theme_data, temperature, key, embedding_model, tokenizer):
    # data : 뉴스 데이터
    # theme_data : 기존 테마명 데이터
    # temperature : GPT API의 답변 온도 조정. 높을수록 창의적인 답변을 많이 하나 불안정함.
    # key : API key 제공
    # embedding_model : 한국어 단어 또는 문장 임베딩 모델. 기본적으로 KR-SBERT 사용
    # theme_embedding : 기존에 존재하는 테마명 리스트에 대해 임베딩 된 값
    # tokenizer : 토큰 수를 셀 수 있는 tokenizer. 여기선 GPT-3.5-turbo 토크나이저 사용

    # 기존 테마명 임베딩
    theme_embedding = embedding_model.encode(theme_data)

    # 명령문 프롬프트
    order_prompt = """[뉴스 기사]를 분석하여 주식 시장에 영향을 미칠 만한 내용인지 판단한 후, 
  가장 관련이 높은 주식 테마 키워드 한 단어와 해당 테마의 연관어 다섯 가지를 추출하세요. 
  테마명 추출 예시로는 [기존 테마명 리스트]를 참고하세요."""

    for idx, (title, text) in enumerate(zip(data["TITLE"], data["CONTENT"])):
        if data["THEME"][idx] != "":
            continue

        if pd.isna(text):
            continue

        # 뉴스 제목 임베딩
        title = data.TITLE[idx]
        title_embedding = embedding_model.encode(title)

        # 기존 테마명과 뉴스 제목의 유사도 계산
        similarities = util.cos_sim(title_embedding, theme_embedding)
        similarities = similarities[0].tolist()

        # 뉴스 제목과 유사도가 높은 순으로 100개의 테마명 추출
        sorted_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )
        selected_indices = sorted_indices[:100]
        theme_selection = [theme_data[i] for i in selected_indices]

        # 기존 테마명 리스트 제공 프롬프트
        theme_prompt = "[기존 테마명 리스트] \n [" + ", ".join(theme_selection) + "]"

        prompt_default = [
            {
                "role": "system",
                "content": """뉴스 기사의 주요 주식 테마 키워드를 추출하는 프로그램입니다. 이 프로그램은 다음과 같은 단계로 진행합니다.
                    STEP 1. 뉴스 기사가 주식 시장에 영향을 미칠 만한 내용을 담고 있는지 판단하세요.
                    STEP 2. 그렇지 않다고 판단될 경우, '주식 연관성: 없음'을 출력하고 종료하세요.
                    STEP 3. 그렇다고 판단될 경우, '주식 연관성: 있음'을 출력합니다.
                    STEP 4. 뉴스 기사와 가장 연관 있는 테마명을 하나 추출합니다. (기존 테마명 리스트에 존재하지 않는 단어도 추출 가능합니다.)
                    STEP 5. 기사 내의 정보를 바탕으로 해당 테마명의 연관어 5가지를 추출합니다.
                    STEP 6. (출력) '주식 연관성: {}, 주식 테마 키워드: {}, \n 연관어: {}' 형태로 출력합니다.
                    
                    이 프로그램은 다음 요구사항을 준수합니다.
                     - 기사 내의 정보에만 의존하여 결과를 제공합니다.
                     - 단순히 기사의 주제가 아닌 주식 시장에 영향을 미칠 만한 주식 테마 키워드만을 출력합니다.
                     - 여러 테마 키워드가 감지될 경우, 가장 확률이 높은 테마 키워드 하나만 출력합니다.
                     - 공시자료에서 사용하는 단어를 연관어로 출력합니다.""",
            },
            {"role": "user", "content": theme_prompt},
            {
                "role": "assistant",
                "content": """기존 테마명 리스트를 확인하였습니다. 
                      기존 테마명 리스트에 존재하지 않는 단어도 테마로 추출 가능함을 인지합니다.""",
            },
        ]

        # 텍스트 길이의 토큰이 GPT 제한 초과시 CLOVA SUMMARY API를 통해 뉴스 기사를 10문장으로 요약
        if len(tokenizer.encode(text)) < 2500:
            text = text
        else:
            text = clova_news_summary(text, language="ko", tone=3, summaryCount=10)

        # 프롬프트 완성
        news_prompt = [
            {"role": "user", "content": order_prompt + " \\n [뉴스 기사] \\n" + text}
        ]
        assitant_prompt = [{"role": "assistant", "content": "주식 연관성: "}]
        prompt = prompt_default + news_prompt + assitant_prompt

        openai.api_key = key  # API Key

        # GPT API 호출
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=prompt, temperature=temperature
        )
        data["THEME"][idx] = completion["choices"][0]["message"]["content"]
        print(str(idx) + completion["choices"][0]["message"]["content"])
        time.sleep(1)

    return data


######################### themeGPT2 : 통합


def generate_word_group(theme_list, embedding_model, threshold):
    # theme_list : 테마 리스트
    # embedding_model : 임베딩용 모델 (사전학습된 모델 또는 학습한 모델)
    # threshold : 테마 그룹을 묶을 유사도 기준

    # 테마가 뽑힌 경우만 가져오기
    themes = [item for item in theme_list if item != "없음" and pd.notna(item)]
    vectors = embedding_model.encode(themes)

    # 코사인 유사도 행렬 계산
    cosine_sim_matrix = cosine_similarity(vectors)

    # 임계값 이상인 인덱스 추출
    indices = np.where(cosine_sim_matrix >= threshold)

    # 유사한 단어들을 집합(set)으로 묶기
    word_groups = []
    for i, j in zip(*indices):
        if i != j:
            group = {themes[i], themes[j]}
            word_groups.append(group)

    # 중복 제거 및 동일 인덱스가 있는 set 합치기
    unique_word_groups = []
    used_indices = set()
    for group in word_groups:
        if not group & used_indices:  # 겹치는 인덱스가 없는 경우
            unique_word_groups.append(group)
            used_indices.update(group)
        else:  # 겹치는 인덱스가 있는 경우, 기존의 그룹에 합치기
            for existing_group in unique_word_groups:
                if existing_group & group:
                    existing_group.update(group)
                    used_indices.update(group)
                    break

    final_word_groups = [group for group in unique_word_groups if len(group) > 1]
    final_word_groups = [
        s for s in final_word_groups if not all(item.isupper() for item in s)
    ]

    return final_word_groups


def themeGPT2(word_groups, temperature, key):
    # word_groups : 유사도가 높은 테마명 그룹, 리스트로 제공
    # temperature : GPT temperature
    # key : OPNEAI API key

    theme_combine_prompt = "아래 테마명 리스트에 대해 테마명 통합 프로그램을 시행하여 결과를 출력하세요."

    prompt_default = [
        {
            "role": "system",
            "content": """테마명 리스트를 받았을 때, 리스트를 하나의 테마명 단어로 통일하는 프로그램입니다. 이 프로그램은 다음 요구사항을 준수합니다.

 - 출력값 예시) "통일한 단어: 2차전지"
 - 단어 외의 값은 출력하지 마세요.
 """,
        }
    ]

    openai.api_key = key
    combine_word_list = []
    for themes in word_groups:
        prompt = prompt_default + [
            {"role": "user", "content": theme_combine_prompt + " " + str(themes)},
            {"role": "assistant", "content": "통일한 단어: "},
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=prompt, temperature=temperature
        )
        combine_word_list.append(completion["choices"][0]["message"]["content"])

    combine_word_list = [
        word.replace("통일한 단어: ", "").replace('"', "").replace("'", "")
        for word in combine_word_list
    ]

    def remove_text_after_second_comma(input_text):
        # 문자열에서 두 번째 쉼표의 위치를 찾습니다.
        first_comma_index = input_text.find(",")

        # 첫 번째 쉼표를 찾을 수 없거나 쉼표가 두 번 이상 나타나지 않으면 입력 텍스트를 그대로 반환합니다.
        if first_comma_index == -1 or input_text.count(",") < 2:
            return input_text

        # 두 번째 쉼표의 위치를 찾습니다.
        second_comma_index = input_text.find(",", first_comma_index + 1)

        # 두 번째 쉼표 이후의 텍스트를 제거합니다.
        result_text = input_text[:second_comma_index]

        return result_text

    combine_word_list = [
        remove_text_after_second_comma(text) for text in combine_word_list
    ]

    return combine_word_list


######################### themeGPT3 : 요약


# AI의 요약 작성 함수
def AI_summary(theme, news, key, temperature):
    # theme : 테마명
    # news: 해당 테마명과 관련된 뉴스 데이터
    # key : OPENAI API key
    # temperature : GPT temperature

    summary_news = [
        clova_news_summary(text, language="ko", tone=2, summaryCount=5) for text in news
    ]

    order_prompt = (
        '아래 리스트로 된 뉴스 기사 3가지를 보고 "' + theme + '" 테마에 대한 이슈와 전망을 5문장 이내로 요약하세요.'
    )

    prompt = [
        {
            "role": "system",
            "content": """
                    - 뉴스 기사의 내용을 바탕으로 5문장 이내로 요약합니다. 
                    - 각 문장은 하오체로 작성합니다. 
                    - 한 문단으로 문장을 이어서 작성합니다.""",
        },
        {"role": "user", "content": order_prompt + "\\n" + str(summary_news)},
    ]

    openai.api_key = key  # API Key
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, temperature=temperature
    )

    theme_summary = completion["choices"][0]["message"]["content"]

    return theme_summary


# AI로 정하는 테마 클러스터 명칭
def AI_cluster_name(theme, key, temperature):
    # theme : 같은 클러스터에 속한 테마명 리스트
    # key : API key
    # temperature : GPT temperature

    order_prompt = "다음 테마명들로 구성된 테마 클러스터에 대한 명칭을 정하세요. " + str(theme)

    prompt = [
        {
            "role": "system",
            "content": """
                    - 클러스터에 속하는 테마들을 포괄할 수 있는 클러스터 명칭을 반환합니다.
                    - 출력 형식은 "클러스터명 : (3개 이하의 단어)"입니다. 
                    - 클러스터 명칭만을 출력하고 다른 말은 출력하지 않습니다.""",
        },
        {"role": "user", "content": order_prompt},
        {"role": "assistant", "content": "클러스터명 : "},
    ]

    openai.api_key = key  # API Key
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, temperature=temperature
    )
    cluster_name_answer = completion["choices"][0]["message"]["content"]

    return cluster_name_answer
