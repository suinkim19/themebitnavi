#!/usr/bin/env python3
# -*- codig: utf-8 -*-
import requests
import json

client_id = "CLOVA SUMMARY API에서 발급받은 클라이언트 ID"
client_secret = "CLOVA SUMMARY API에서 발급받은 클라이언트 PW"


def clova_news_summary(content, language, tone, summaryCount):
    # content : 뉴스기사 내용에 대한 텍스트
    # langauge : 언어 설정. kor : 한국어
    # tone : CLOVA summary tone
    # sumamryCount : 요약 문장 개수

    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/json",
    }
    url = "https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize"
    data = {
        "document": {"content": content},
        "option": {
            "language": language,
            "model": "news",
            "tone": tone,
            "summaryCount": summaryCount,
        },
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    rescode = response.status_code
    if rescode == 200:
        return clova_cleanning(response.text)
    else:
        return "Error : " + response.text


def clova_cleanning(text):
    # text : CLOVA summary API를 이용하여 요약된 텍스트 출력

    # '{"summary":"', "}' 사이의 텍스트 추출
    start_index = text.find('{"summary":"') + len('{"summary":"')
    end_index = text.find('"}', start_index)
    extracted_text = text[start_index:end_index]

    # 작은 따옴표, 줄바꿈표 제거
    cleaned_text = extracted_text.replace("'", "").replace("\\n", " ").replace("\\", "")

    return cleaned_text
