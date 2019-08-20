# Introduction  
Pokemon 이미지 800여 장으로 DCGAN 모델을 구현해봅니다.  
데이터세트는 Kaggle의 잘 정리된 포켓몬 정사각형 사진을 사용하였습니다.  
소스코드는 실행 할때, 구글 Colab 환경에서 작동하시길 권유합니다.  
1. 구글 드라이브에 직접 데이터셋을 업로드하기  
2. 캐글 API로 구글 코랩과 연동후 데이터셋 다운로드  
3. 캐글 커널에서 데이터셋 다운로드 후 실행  
    
저는 데이터셋이 구글 드라이브에 로드되어 있어 바로 마운트하고 사용하였습니다.  
데이터셋 링크 : https://www.kaggle.com/kvpratama/pokemon-images-dataset  
    
***
# DCGAN with Pokemon Summary    
Pokemon 이미지를 활용하여 DCGAN을 구현하였습니다.  
    
### High Resolution  
![High1](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/High%20Resolution%20Sample/High%20Resolu%207.png)  
![High2](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/High%20Resolution%20Sample/High%20Resolu%206.png)  
![High3](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/High%20Resolution%20Sample/High%20Resolu%204.png)  
    
### Low Resolution  
![sample2](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/Low%20Resolution%20Sample/Low%20Resolu%20(2).png) 
![sample4](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/Low%20Resolution%20Sample/Low%20Resolu%20(4).png)  
![sample6](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/Low%20Resolution%20Sample/Low%20Resolu%20(6).png)
