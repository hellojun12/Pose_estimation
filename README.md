# Pytorch pose-estimation



## 📍 Dataset
--------
1. 지원 dataset
    - lsp
  
<br>

1. image data 삽입 경로
   
    -  data/{데이터셋 이름}/images  
<br>

2. annotation data 삽입 경로(.mat 파일)
    -  data/{데이터셋}/{데이터셋 이름}.mat  

ex)
```
data

  ↳  data/lsp/images
    ↳ im0001.jpg
    ↳ im0002.jpg
         ...

  ↳ data/lsp/joints.mat
```
<br>

### 📍 Model 설정

1. 지원 모델
   - Stacked hourglass (hg)
   - HRNet (hrnet)  
<br>

2. 모델 설정
   -  main.py 파일, parse부분 설정  
        ``` 
        python main.py --architecture {hg, hrnet}
        ```
  

### 📍 Heat map encoding/decoding
1. 지원 
   - regular gaussian heatmap
   - DARK  
<br>

2. 설정
    - main.py 파일, parse부분 설정
        ```
        python main.py --heatmap {regular, dark}
        ```

### 📍 Quick Start

```
python main.py --dataset lsp -architecture hg --stacks 8 --blocks 1
```
