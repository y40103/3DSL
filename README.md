# 3DSL 

StructureLight 3D SCAN 
基於結構光的3D掃描


## Testing Environment

Linux Mint 21
Python3.10


## Installation

```
pip install -r requirements.txt

```


## Run it

```
python3 graycode_1204_numba.py

```



## 隨筆

蠻久以前寫的功能  
那時花了不少心思 或許會有人需要也說不定  
如果對掃描這塊有興趣 或許可以參考看看  

由於當時掃描這方面似乎沒有library可以用  
有相當一部份密集運算是用python實現  
效能上其實不佳, 之前有用numba加速  
但似乎是numba更新或是OS問題  
在Linux Mint 21 python3.10 使用numba加速會有一些issue, 測過3.7 問題依然存在  
以前是在windows上開發 現在手邊也沒有windows的系統可以測試  
實際上我也忘記之前windows開發使用的python版本  
因此就先將numba取消 至少是可以正常運行的  

*.pkl 是校正數據  
主要為投影機的校正與馬達旋轉軸心的校正  
原理是參考  
https://ieeexplore.ieee.org/document/6375029  
源碼當時實現的亂七八糟 就不拿出來了  

之前掃描是有配合機電運動旋轉 藉由4個面拼成主體  

輸出檔為三維空間的座標資訊  

可由第三方軟體建成模型  
