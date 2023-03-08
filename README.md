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

#### ※ 전처리 후 : 21개 컬럼 59,400개 행 데이터셋 
![image](https://user-images.githubusercontent.com/110115061/221491352-424caf71-109d-4049-8d1e-186fd2f4613b.png)  
- 타겟 컬럼 : status (수자원의 작동여부, 수리 필요 여부)



---
<br/>

# 3. 모델 구현
## 3.1. 수자원 상태 예측에 대한 몇 가지 추론과 가정
### 3.1.1. 수자원의 물의 양과 품질이 타겟 값 예측에 중요한 영향을 미치지 않을까?
![image](https://user-images.githubusercontent.com/110115061/221492378-ee80988e-f472-4ff9-be93-0a7f0d15b6fc.png)  
- countlpot으로는 quantity 컬럼의 dry값을 제외하고 확실한 영향을 확인할 수 없었음
- 모델링 이후 특성 중요도를 통해 확인 필요

### 3.1.2. 수자원 이용료 관련, 무료로 이용하는 수자원들이 문제가 많지 않을까?
![image](https://user-images.githubusercontent.com/110115061/221492719-4f544b7c-82f9-4559-ab7c-474ea617950b.png)  
- annually, per bucket, monthly 단위로 이용료는 내는 수자원이 잘 작동하는 비율이 높았음
- 이용료가 없거나 지출방식이 불분명한 경우 작동불가한 수자원이 더 많은 것을 확인할 수 있음

#### ※ 수자원의 유지보수와 더불어 이용자의 자립의식 고취를 위해서도 **소정의 이용료를 부과**하는 방식이 공익을 위해 중요함
#### ※ 향 후 수자원을 수리하거나 재설치 차 방문하는 경우, 설치 및 유지보수 뿐 아니라 **커뮤니티 차원의 우물관리교육을 함께 진행**하고 실행여부를 모니터링하는 것이 좋을 듯 함

### 3.2.3. 오래된 수자원일수록 작동이 되지 않는 경우가 많지 않을까?  
![image](https://user-images.githubusercontent.com/110115061/223608330-95d6b9b1-162e-4f3d-95e6-b651db2d11c8.png)
- 'unknown'의 경우 정확한 설치년도를 알 수 없음
- 예상대로 오래된 수자원일수록 작동불가의 미율이 더 높고, 최근에 설치했을수록 잘 작동하는 비율이 월등히 높음
- **각 지역별 오래된 수자원들의 상태조사를 별도로 실시**하는 것도 유지보수차원에서 좋은 대처 방안이 될 것으로 보임

<br/>

## 3.2. 모델링 전 전처리
- 타겟 클래스 확인 후 데이터를 목적에 따라 분할  
![image](https://user-images.githubusercontent.com/110115061/223610579-755aba49-177d-49b1-bc3d-62e64f2ff23d.png)  
- 학습용, 검증용, 평가용 데이터 분할, 학습용 데이터의 타겟 클래스 비율은 분할 전과 동일하게 유지  
![image](https://user-images.githubusercontent.com/110115061/223610751-ff70ea09-55d4-4408-85fe-5f17fdcd7cf8.png)  
- 타겟 컬럼과 특성컬럼 X,y로 분할

<br/>

## 3.3. 모델링
### 3.3.1. 기준모델 : Random Rate Classifier
![image](https://user-images.githubusercontent.com/110115061/223614316-bf33026c-58a5-4c1b-ab5d-4b6ee4fa1586.png)   
- 기준모델은 앞으로 구현할 모델 성능의 최소 기준을 의미함
- 기준모델의 정확도는 44.786%

### 3.3.2. RandomForestClassifier
![image](https://user-images.githubusercontent.com/110115061/223614517-2c2e43df-8305-48a2-aea8-969cab163986.png)  
- 집단지성의 긍정적 효과를 얻을 수 있는 RandomForest모델은 간단한 구현에도 불구하고 좋은 성능을 보여주며, 다중분류 문제에도 별다른 설정이 필요없이 모델링이 가능함  
![image](https://user-images.githubusercontent.com/110115061/223614687-29e937ee-3d3c-4901-a7ab-46cc4579c486.png)  
- 모델의 성능을 높일 수 있는 RandomizedSearchCV 를 통해 모델의 성능에 영향을 주는 설정값을 조정함, 
  이때 여러 설정값의 **랜덤한 조합을 적용**하고, 교차 검증하여  매 조합마다 점수(정확도)를 매겨 성능을 평가함

### 3.3.3. XGBClassifier
![image](https://user-images.githubusercontent.com/110115061/223614979-f277481a-bd0f-4c96-83dc-add8aa8cf690.png)  
- 기존 모델의 잔차(오차)를 학습하여 개선된 모델을 만들어가는, 비유하자면 복습에 능한 모델  
![image](https://user-images.githubusercontent.com/110115061/223615063-dc473999-a3bc-4ac3-9e40-f0d4b8976564.png)  
- 모델의 성능을 높일 수 있는 RandomizedSearchCV 를 적용

### 3.3.4. LightGBM
![image](https://user-images.githubusercontent.com/110115061/223615482-79a7e426-cad3-4576-b6e7-7d1d54245af4.png)  
- XGBoost처럼 복습에 능한 모델이지만 학습 및 예측시간이 비교적 상당히 빠른 편이고 메모리도 더 적게 사용함
- 다른 모델보다 시간도 훨씬 빠르고, 성능 또한 준수하기 때문에 바로 Optuna로 설정값 조정 진행
![image](https://user-images.githubusercontent.com/110115061/223615608-b79c1152-3349-4fa3-939a-13c13e829e67.png)  
- Optuna는 이전 설정값의 성능보다 더 좋은 설정값을 알아서 찾아주는 오픈소스툴

<br/>

## 3.4. 모델 평가
### 3.4.1. 성능 평가 및 최종모델 선정
- classification report
![image](https://user-images.githubusercontent.com/110115061/223616155-649b73a5-5523-4c11-a9ea-04a1ecde5347.png)  

- 혼동행렬
![image](https://user-images.githubusercontent.com/110115061/223618945-63d9bba9-3c53-4538-a1bc-14e09f936623.png)  

#### ※ 최종모델은 LightGBM모델에 Optuna로 최적화한 모델로 선정함
- 학습하거나 타겟값을 예측하는 시간이 다른 모델보다 상당히 빠름
- 빠른 시간 대비 높은 성능을 보여줌
- 따라서 추가작업, 수정, 보완 등이 원활할 것으로 보임
- 하이퍼파라미터 최적화 과정 시각화  
![image](https://user-images.githubusercontent.com/110115061/223627249-0350fac9-6c93-4cce-9885-f34737b7db5f.png)  

### 3.4.2. 최종모델 평가
- 평가용 데이터 적용  
![image](https://user-images.githubusercontent.com/110115061/223627390-e4cd1cff-f09c-4d06-a743-cb4e7120937b.png)  
- 약 79%의 정확도
- 수리 필요 여부 판단의 F1 score가 다소 떨어짐
- 특성 순열중요도  
![image](https://user-images.githubusercontent.com/110115061/223627568-26d659a8-5449-4c08-aa9f-821999560e7d.png)  
- 3.1.1. 물의 양과 품질이 모델 타겟 예측에 중요한 영향을 미치는지 확인
  - 모델의 타겟 예측에는 예상대로 quantity의 영향이 컸음, 반면 quality의 영향은 작은 것을 알 수 있음
  - 경험자의 이야기에 따르면 수자원을 이용하는 주민들은 물이 워낙 귀하다 보니 물의 품질에 대해 그리 까다롭게 굴지 않는다고 함



---
<br/>

# 4. 결론
- 데이터를 잘못 예측하는 경우가 아직 많기 때문에 최종모델 단독으로 수자원의 상태를 예측하는 것은 무리가 있음
- 타겟 예측에 결정적인 컬럼 위주로 데이터셋을 재구성하거나 수가 적은 타겟 클래스의 데이터를 보충하는 등 데이터와 모델 전반으로 추가적인 수정 및 보완이 필요해 보임
- 하지만 ML모델을 함께 사용하여 탄자니아 전 지역의 우물 상태를 판단하는 것은 유용함
  - 활용 예시 : 수리가 필요한 water point의 gps좌표 활용(장비운용계획, 스케줄 조율 등)  
  ![image](https://user-images.githubusercontent.com/110115061/223628042-8200efd0-60d3-4e56-9317-b35a9e2c9b0e.png)  


