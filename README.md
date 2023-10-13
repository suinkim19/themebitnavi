## 주식 테마 탐지 및 테마 네트워크 시각화 서비스

---

ThemeGPT.py : ThemeGPT를 통한 테마 추출, 테마명 통합, 테마명 요약에 대한 함수 파일입니다. 

theme_index.py : 테마지수 계산에 사용되는 함수 파일입니다. 

theme_assess.py : ThemeGPT 성능 평가에 사용된 함수 파일입니다. 

network_function.py : 테마 네트워크 구축 및 시각화에 사용된 함수 파일입니다. 

clova_summary.py : CLOVA summary API 사용에 사용된 함수 파일입니다. 

article_processing.py : 뉴스 데이터의 전처리에 사용된 함수 파일입니다. 

ThemeGPT.ipynb : 테마 추출, 테마명 통합, 테마명 요약 등 GPT API를 사용하는 모든 모델에 대한 예시입니다. 

theme_stock_and_index.ipynb : GPT가 추출한 각 테마에 속한 종목들을 선정하고, 테마지수를 계산하는 예시입니다. 

theme_network.ipynb : 테마 네트워크를 구축하고 시각화하는 코드의 예시입니다. 

---

### 테마 네트워크 시각화 예시
![테마빛나비1](https://github.com/suinkim19/themebitnavi/assets/103398227/c44571ed-c2f9-4cc1-a2c5-5c2e78e51275)
![테마빛나비2](https://github.com/suinkim19/themebitnavi/assets/103398227/3a29e908-510d-4823-8f85-9380fd63c15c)

### API 발급 리스트

### DART Open API 키 발급하기

1. https://opendart.fss.or.kr/ 접속 
2. 인증키 신청/관리 -> 인증키 신청
3. 계정 생성 및 API키 발급
4. 공시자료_crawl.ipynb에서 API 키를 입력

### GPT API 키 발급하기

1. https://platform.openai.com/ 접속
2. 로그인 후 API로 이동
3. 오른쪽 상단 Personal 클릭 후 View API key에 접속하여 API key 발급
4. ThemeGPT.ipynb에서 API 키를 입력

### CLOVA SUMMARY API 키 발급하기

1. https://console.ncloud.com/dashboard에 로그인 후 접속
2. 콘솔에서 Services > AI, NAVER API > Application에서 Application 등록
3. CLOVA Summary API를 발급하고, 인증 정보에서 Client ID 및 Password 확인
4. clovasummary.py에서 ID, password를 입력
