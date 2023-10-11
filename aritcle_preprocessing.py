import re
from bs4 import BeautifulSoup
# IMPORTANCE >= 60인 기사들 필터링 함수
def filter_dataframe(df):
    # IMPORTANCE가 60 이상인 행 필터링
    filtered_data = df[df['IMPORTANCE'] >= 60]

    # 필요한 열만 선택
    article = filtered_data[['DATE_TIME', 'WRITER', 'TITLE', 'IMPORTANCE', 'ITEM_NAME', 'TAG_LIST', 'CONTENT']]

    # 인덱스 재정렬
    article = article.reset_index(drop=True)

    # 날짜 재설정
    article['DATE_TIME'] = article['DATE_TIME'].str[2:8]
    article = article.rename(columns={'DATE_TIME': 'DATE'})
    
    return article


# 뉴스 본문 1차 전처리 함수
def clean_content(html_content):
    cleaned_text = re.sub(r'<.*?>', '', html_content)
    
    soup = BeautifulSoup(html_content, 'html.parser')
    # 스크립트와 스타일 태그 제거
    for script in soup(["script", "style"]):
        script.decompose()

    cleaned_text = soup.get_text()
    cleaned_text = '\n'.join([line.strip() for line in cleaned_text.splitlines() if line.strip()])
    cleaned_text = cleaned_text.replace("_x000D_\n", " ")

    return cleaned_text

# CEO 스코어데일리
def scoredaily(content) :
    pattern = r'\[CEO스코어데일리 / .* 기자 / .*@.*\]'
    main_content = re.split(pattern, content)[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# CNBC(AI뉴스)
def cnbc(content) :
    pattern = "AI 분석&요약AI 뉴스분석 안내 AI 모델을 활용하여 뉴스의 중요도와 뉴스에 나타난 심리를 보여드립니다.   사용 AI 모델  - 중요도 모델: 애널리스트가 투자에 중요하다고 생각하는 뉴스 데이터와 조회수가 높은 뉴스 데이터를 바탕으로 '중요 뉴스'를 판단하도록 학습한 인공지능 모델로 뉴스의 중요도를 계산합니다.  - 긍부정 모델: 네이버 긍부정 분류 AI 모델을 통해 뉴스에 나타난 심리를 예측합니다. 중요도 "
    pattern2 = r'Original Article.*'
    main_content = content.replace(pattern, '')
    main_content = re.sub(pattern2, '', main_content)
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# IR큐더스
def ircuders(content) :
    pattern = "[IR브리핑] "
    pattern2 = "IR Page에 접속하시면 다양한 투자 정보를 확인할 수 있습니다.바로가기 >> \xa0 본 내용은 기업에 대한 이해 증진을 위한 목적으로, 투자 권유를 목적으로 한 것이 아닙니다.투자에 관한 결정은 투자자 본인에게 있으며, 회사는 투자에 관해 일체의 책임을 지지 않습니다."
    main_content = content.replace(pattern, '').replace(pattern2, '').replace('\xa0',' ')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# NSP통신
def nsp(content) :
    pattern = r'\(서울=NSP통신\) .* 기자 = '
    pattern2 = r"\nNSP통신 .+ 기자\(.+@.+\.com\)\nⓒ.+ NSP통신·NSP TV. 무단전재-재배포 금지\."
    main_content = re.sub(pattern, '', content)
    main_content = re.sub(pattern2, '', main_content)
    main_content = main_content.replace('\n',' ')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# SCMP(AI뉴스)
def scmp(content) :
    pattern = "AI 분석&요약AI 뉴스분석 안내 AI 모델을 활용하여 뉴스의 중요도와 뉴스에 나타난 심리를 보여드립니다.   사용 AI 모델  - 중요도 모델: 애널리스트가 투자에 중요하다고 생각하는 뉴스 데이터와 조회수가 높은 뉴스 데이터를 바탕으로 '중요 뉴스'를 판단하도록 학습한 인공지능 모델로 뉴스의 중요도를 계산합니다.  - 긍부정 모델: 네이버 긍부정 분류 AI 모델을 통해 뉴스에 나타난 심리를 예측합니다. 중요도 "
    main_content = content.replace(pattern, '')
    main_content = main_content.split('기사 본문(원문)')[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# eDaily
def edaily(content) :
    pattern = r'\[이데일리 [가-힣 ]{3} 기자\]'
    pattern2 = r'▶▶.*＜ⓒ종합 경제정보 미디어 이데일리 - 무단전재 & 재배포 금지＞'
    main_content = re.sub(pattern, '', content)
    main_content = re.sub(pattern2, '', main_content)
    main_content = main_content.replace('_x000D_',' ')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 글로벌이코노믹
def globaleco(content) :
    pattern = r'([가-힣\s]+)\s(글로벌이코노믹\s.*기자)\s([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7})'
    main_content = re.sub(pattern, '', content)
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 뉴 프라임경제
def newprime(content) :
    pattern = "[프라임경제] "
    main_content = content.replace(pattern, '').replace('\xa0',' ')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 뉴스웨이
def newsway(content) :
    pattern = r'([가-힣\s]+)기자\s([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7})'
    pattern2 = r'<온라인 경제미디어 뉴스웨이 - 무단전재 & 재배포 금지>'
    pattern3 = r'([가-힣\s]+)기자'
    main_content = re.sub(pattern, '', content)
    main_content = re.sub(pattern2, '', main_content)
    main_content = re.sub(pattern3, '', main_content)
    main_content = main_content.replace('그래픽=','').replace('사진=', '').replace('\\n','').replace('\n','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 뉴스핌
def newspim(content) :
    pattern = r'\[(.*?)=(.*?)\]\s(.*?)='
    pattern2 = r'\[(.*?)=(.*?)\]\s(.*?)\s='
    pattern3 = r'\[사진=(.*?)\]'
    pattern4 = r'\[사진\s=(.*?)\]'
    pattern5 = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}'
    pattern6 = r'이 기사는.*$'
    pattern7 = r'저작권자\(c\).*?무단 전재-재배포 금지'
    
    main_content = re.sub(pattern, '', content)
    main_content = re.sub(pattern2, '', main_content)
    main_content = re.sub(pattern3, '', main_content)
    main_content = re.sub(pattern4, '', main_content)
    main_content = re.sub(pattern5, '', main_content)
    main_content = re.sub(pattern6, '', main_content)
    main_content = re.sub(pattern7, '', main_content)
    main_content = main_content.replace('\xa0',' ')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 로이터(AI뉴스)
def reuter(content) :
    pattern = "AI 분석&요약AI 뉴스분석 안내 AI 모델을 활용하여 뉴스의 중요도와 뉴스에 나타난 심리를 보여드립니다.   사용 AI 모델  - 중요도 모델: 애널리스트가 투자에 중요하다고 생각하는 뉴스 데이터와 조회수가 높은 뉴스 데이터를 바탕으로 '중요 뉴스'를 판단하도록 학습한 인공지능 모델로 뉴스의 중요도를 계산합니다.  - 긍부정 모델: 네이버 긍부정 분류 AI 모델을 통해 뉴스에 나타난 심리를 예측합니다. 중요도 "
    main_content = content.replace(pattern, '')
    main_content = main_content.split('기사 본문(원문)')[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 데이터투자
def datainvest(content) :
    pattern = r'[가-힣 ]{3} 데이터투자 기자'
    main_content = re.split(pattern, content)[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 뉴시스
def newsis(content) :
    pattern = '◎공감언론'
    main_content = re.split(pattern, content)[0].strip()
    try:
      main_content = re.split('=', main_content)[2]
    except:
      pass
    main_content = main_content.replace('\\n', '').replace('\n','').replace('◆','').replace('  ', '')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 매일경제
def maeil(content) :
    pattern = r'\[([가-힣\s]+)\s([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7})\]'
    pattern2 = r'\[ⓒ[^\]]+\]'
    main_content = re.sub(pattern, '', content)
    main_content = re.sub(pattern2, '', main_content)
    main_content = main_content.replace('`\`', '').replace('_x000D_','').replace('  ', '')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 머니투데이
def moneytoday(content) :
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    main_content = re.split(pattern, content)[0].strip()
    split_part = re.split('\.', main_content)
    main_content = '.'.join(split_part[:-1])
    main_content = re.sub(r'\[머니투데이[^\]]+기자\]', '', main_content)
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content + '.'

# 머니투데이방송
def mtn(content):
    main_content = re.split('머니투데이방송 MTN 기자', content)[0].strip()
    split_part = re.split('\.', main_content)
    main_content = '.'.join(split_part[:-1],)
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content + '.'

# 서울경제
def seouleco(content):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    main_content = re.split(pattern, content)[0].strip()
    split_part = re.split('\.', main_content)
    main_content = '.'.join(split_part[:-1],)
    main_content = main_content.replace('\\n', '').replace('\n','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content + '.'

# 아시아경제
def asiaeco(content):
    pattern = '●'
    main_content = re.split(pattern, content)[0].strip()
    main_content = main_content.replace('\\n', '').replace('\n','')
    main_content = main_content.replace('`\`','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 연합뉴스
def yeonhab(content):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    main_content = re.split(pattern, content)[0].strip()
    main_content = re.split('=', main_content)[2]
    main_content = main_content.replace('\\n', '').replace('\n','').replace('  ', '')
    main_content = re.sub(r'\\', '', main_content)
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 한경뉴스
def hankyung(content) :
    pattern = '([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
    main_content = re.split(pattern, content)[0].strip()
    
    main_content = main_content.replace('한국경제 & hankyung.com, 무단전재 및 재배포 금지','')
    main_content = main_content.split('본 글은 투자 참고용')[0].strip()
    main_content = main_content.replace('ⓒ','').replace('▼','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 조선경제
def chosun(content) :
    pattern = r'[가-힣 ]{3} 기자.*?$'
    main_content = re.split(pattern, content)[0].strip()
    main_content = main_content.replace('\\n','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 이투데이
def etoday(content) :
    # 기자 이메일 기준 제거
    main_content = content.split('@etoday.co.kr)]')[1].strip()
    
    # 기사 리스트 제거
    main_content = main_content.split('[종목기사]')[0].strip()
    main_content = main_content.split('[오늘의 핫클릭]')[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 컨슈머타임스
def consumer(content) :
    pattern = r".*?(기자\s\|\s)"
    main_content = re.sub(pattern, "", content).strip()
    main_content = main_content.replace('\\n','').replace('\\r','').replace('\n','').replace('\r','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 헤럴드경제
def herald(content) :
    pattern = r".*?(\[헤럴드경제=.*\s기자\])"
    main_content = re.sub(pattern, "", content).strip()
    main_content = main_content.split('▶ 무조건 오르는 비공개 급등종목')[0].strip()
    main_content = main_content.replace('\n','')
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content 

# 파이낸셜뉴스
def financial(content) :
    pattern = '([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
    main_content = re.split(pattern, content)[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content

# 인포스탁
def infostock(content) :
    pattern = '([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+(\.[A-Z|a-z]{2,})+'
    main_content = re.split(pattern, content)[0].strip()
    main_content = main_content.split('Update')[0].strip()
    main_content = re.sub(r'\s+', ' ', main_content)
    return main_content