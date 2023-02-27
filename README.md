# 프로젝트2 : 탄자니아 수자원 현황 예측 머신러닝 모델 개발

# 1. Overview

## 1.1. 배경과 목적
- 다수의 아프리카 국가들이 처해 있는 물부족 및 오염 문제를 해결하기 위해 마을 곳곳에 설치된 우물(수자원)
- 우물은 유지보수가 굉장히 중요한데, 소홀한 관리로 물이 오염되거나 수원이 막히는 경우 마을 주민들의 생활과 생존에 큰 피해가 발생함
- 아프리카의 한정된 인적, 물적 자원으로 지역사회 생명의 역할을 하는 우물을 보다 효율적으로 관리할 수 있는, 수자원의 상태를 예측하는 모델 개발

## 1.2. 데이터 소개
- 탄자니아 수자원부의 데이터를 집계하는 Taarifa waterpoints 대시보드 출처 데이터셋
- 설치 주체, 설치 자금, 관리 주체, 위치, 수자원의 유형과 품질 등 수자원 관리와 관련된 정보
- 총 41개 컬럼 59,400개 행으로 이루어진 데이터

### Columns Description
- id : 해당 우물에 부여된 id
- amount_tsh : 우물 설치에 소요된 자금 (탄자니아 화폐단위)
- date_recorded : 기록 날짜
- funder : 자금 제공자
- gps_height : 우물의 고도
- installer : 우물을 설치한 기관
- longitude : GPS 좌표
- latitude : GPS 좌표
- wpt_name : 워터포인트가 있는 경우 워터포인트의 이름
- num_private :
- basin : 지리적 유역
- subvillage : 지리적 위치
- region : 지리적 위치
- region_code : 지리적 위치(코드)
- district_code : 지리적 위치(코드)
- lga : 지리적 위치
- ward : 지리적 위치
- population : 우물 주변의 인구
- public_meeting : 미팅 여부
- recorded_by : 본 데이터를 입력하는 그룹
- scheme_management : 워터포인트를 운영하는 사람
- scheme_name : 워터포인트를 운영하는 사람의 이름
- permit : 워터포인트가 허용되는 경우
- construction_year : 워터포인트가 건설된 연도
- extraction_type : 워터포인트가 사용하는 추출의 종류
- extraction_type_group : 워터포인트가 사용하는 추출의 종류
- extraction_type_class : 워터포인트가 사용하는 추출의 종류
- management : 워터포인트 관리 방법
- management_group : 워터포인트 관리 방법
- payment : 물 비용
- payment_type : 물 비용
- water_quality : 물의 품질
- quality_group : 물의 품질
- quantity : 물의 양
- quantity_group : 물의 양
- source : 물의 근원
- source_type : 물의 근원
- source_class : 물의 근원
- waterpoint_type : 워터포인트의 종류
- waterpoint_type_group : 워터포인트의 종류
- status : 워터포인트(우물)의 작동여부, 수리 필요 여부

## 1.3. 스킬셋
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- xgboost
- lightgbm
- optuna
- eli5
- pdpbox



---
<br/>

# 2. 데이터 전처리
## Location
- 중요한 지역단위 정보를 제외한 나머지 컬럼 삭제
- 세부적인 위치는 GPS좌표 컬럼의 값을 참고하는 것으로
- GPS좌표를 참고해서 수자원 위치를 그려본 결과 좌상단 이상치 발견, 해당 지역의 GPS 좌표값 평균으로 대체  
![image](https://user-images.githubusercontent.com/110115061/221489486-c83d50c8-0967-44ab-8e6e-a4a92fb85541.png)  

## 기타
- 단일값으로 구성된 컬럼 삭제
- 카디널리티가 높지만 모델링에 필요한 경우 구간화로 값을 정리
  - population(우물 주변의 인구), construction_year(수자원 건축 연도)
  ![image](https://user-images.githubusercontent.com/110115061/221490124-46b1050b-e4bb-45fd-be8f-e2af713ef86a.png)  
  ![image](https://user-images.githubusercontent.com/110115061/221490150-a97e6014-ce5f-4d44-9035-713b552b951f.png)  

### ※ 전처리 후 : 21개 컬럼 59,400개 행 데이터셋 
![image](https://user-images.githubusercontent.com/110115061/221491352-424caf71-109d-4049-8d1e-186fd2f4613b.png)  
- 타겟 컬럼 : status (수자원의 작동여부, 수리 필요 여부)



---
<br/>

# 3. 모델링
## 3.1. 수자원 상태 예측에 대한 몇 가지 추론과 가정
### 수자원의 물의 양과 품질이 타겟 값 예측에 중요한 영향을 미치지 않을까?




- plotly


