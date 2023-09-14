# PA_SASRec

## Description
사용자의 구매 정보와 보조 정보를 활용한 SASRec 성능 향상 프로젝트  

## Environment
CPU : Intel(R) Core(TM) i7-10750H CPU  
GPU : NVIDIA GeForce GTX 1660 Ti  
RAM : 16GB
python version : 3.9.7  
pytorch version : 1.10.0+cu113  

## Model
<img src="https://github.com/chanwoomoon/PA_SASRec/blob/main/img/model.png?raw=true" width="600" height="600"></img>

  > ### Embedding Layer  
  모든 유저 구매기록은 최근 10개로 통일, 구매기록 3개 이하인 사용자 제거  
  각 유저의 구매기록과 보조정보 별로 임베딩을 생성한 후 생성된 구매기록 정보와 보조정보는 보조정보를 기준으로 각각 다른
  입력값으로 사용
  
  
  > ### Parallel multi-head attention  
  이질적인 보조 정보를 독립적으로 활용하기 위해 각 보조정보 별로 하나의 attention block을 병렬적으로 구성
  각 attention block에서 나온 값은 Layer Norm과 Feed forward neural network를 순차적으로 통과하여 하나의 레이어에서 다시 합침
  

## Experiment Setting
Dataset : book_transaction_data  
batch size : 256  
max_len : 10  
epochs : 41  
optimizer : ADAM  
평가지표 : NDCG10, HR10  

## Result

|평가지표|NDCG@10|Improvement|HR@10|Improvement|
|:------:|:---:|:---:|:---:|:---:|
|SASRec|0.377|-|0.512|-|
|PA_SASRec|0.409|<span style="color:red">**8.4%**</span>|0.551|<span style="color:red">**7.6%**</span>|

## run code
``` python
main.py --dataset=book_transactions --train_dir=default --maxlen=10 --dropout_rate=0.2 --device=cuda
```

## Reference
https://github.com/pmixer/SASRec.pytorch
