# Introduction  
Pokemon 이미지 800여 장으로 DCGAN 모델을 구현  
데이터셋은 Kaggle의 잘 정리된 포켓몬 정사각형 사진을 사용  
소스코드는 실행 할때, 구글 Colab 환경에서 작동 권유  
1. 구글 드라이브에 직접 데이터셋을 업로드하기  
2. 캐글 API로 구글 코랩과 연동후 데이터셋 다운로드 (kaggle_colab.ipynb 참조)  
3. 캐글 커널에서 데이터셋 다운로드 후 실행  
    
저는 데이터셋을 다운받아 구글 드라이브에 로드하고 Colab과 연동하여 사용하였습니다.  
데이터셋 링크 : https://www.kaggle.com/kvpratama/pokemon-images-dataset  


# DCGAN for Pokemon Summary  
- Pokemon 이미지를 활용하여 DCGAN을 구현하였습니다.  
    
### High Resolution  
![High1](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/sample1/high%(7).png)  
![High2](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/sample1/high%(6).png)  
![High3](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/sample1/high%(4).png)  
    
### Low Resolution  
![sample2](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/sample2/low%(2).png) 
![sample4](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/sample2/low%(4).png)  
![sample6](https://github.com/Doyosae/GAN_Pokemon/blob/master/DCGAN/sample2/low%(6).png)