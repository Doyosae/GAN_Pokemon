import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.backend.tensorflow_backend import set_session



def Get_Array_Image (Inputs) :
    
    TempTrainImage = []
    
    for filename in Inputs :
        
        TempImage = load_img (ImagePath + "/" + filename, target_size = WantedImageShape [ : 2])
        TempImage = img_to_array (TempImage) / 255
        TempTrainImage.append (TempImage)
        
    Outputs = TempTrainImage
    
    return Outputs


# 데이터셋트를 셔플하고 배치 사이즈만큼 슬라이스해서 내보낸다. 
def Build_Shuffle_BatchData (BatchSize, Input) :
    
    np.random.shuffle (Input)
    TempGetImage = Input[0 : BatchSize]
    Output = TempGetImage
    
    return Output


# 아래에서 사용해야할 변수는 Array로 만든 GetImage를 사용한다.
GetImage = np.array (Get_Array_Image (LoadImage))
print ("포켓몬 캐릭터 사진들의 갯수와 크기, 채널을 출력      ", np.shape(GetImage))

# 이미지를 섞어주는 셔플을 테스트
GetImage = Build_Shuffle_BatchData (100, GetImage)

# Print Image Test
fig, ax = plt.subplots(1, 5, figsize=(20, 20))

for i in range(0, 5) :
    
    ax[i].set_axis_off()
    ax[i].imshow(GetImage[i])

plt.show()


# Generator Function, # Noise가 들어오면 4*4*1024 레이어와 풀로 연결된다.
def Build_Generator (inputs): 
    
    with tf.variable_scope("GeneratorVal"):
        
        output = tf.layers.dense(inputs, 4*4*1024)
        output = tf.reshape(output, [-1, 4, 4, 1024])
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.relu (output)
        
        output = tf.layers.conv2d_transpose(output, 512, [3, 3], strides = (2, 2), padding = "SAME")
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.relu (output)
        
        output = tf.layers.conv2d_transpose(output, 256, [2, 2], strides = (2, 2), padding = "SAME")
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.relu (output)
        
        output = tf.layers.conv2d_transpose(output, 128, [4, 4], strides = (2, 2), padding = "SAME")
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.relu (output)
        
        output = tf.layers.conv2d_transpose(output, 3, [3, 3], strides = (2, 2), padding = "SAME")
        output = tf.tanh (output)
        
    return output
 
    
# Discriminator Function
def Build_Discriminator (inputs, reuse = None):
    
    with tf.variable_scope("DiscriminatorVal") as scope:
        
        if reuse:
            scope.reuse_variables()

        output = tf.layers.conv2d(inputs, 128, [3, 3], strides = (2, 2), padding = "SAME")
        output = tf.nn.leaky_relu (output)
        
        output = tf.layers.conv2d(output, 256, [3, 3], strides = (2, 2), padding = "SAME")
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.leaky_relu (output)
        
        output = tf.layers.conv2d(output, 512, [3, 3], strides = (2, 2), padding = "SAME")
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.leaky_relu (output)
        
        output = tf.contrib.layers.flatten(output)
        output = tf.layers.dense(output, 1, activation = None)
        
    return output
 
    
# Noise Function
def Build_GetNoise (batch_size, noise_size):
    return np.random.uniform(-1.0, 1.0, size=[batch_size, noise_size])



"""
필요한 상수를 선언, 데이터를 재활용할 횟수를 TotalEpoch
한 번 학습할 때마다 쪼갤 데이터 집합을 BatchSize
Generator에 입력될 노이즈의 크기를 NoiseSize
X, Z, IsTraining, global_step에 들어갈 두 변수
"""
TotalEpoch= 100
BatchSize = 100
NoiseSize = 100
LearningRate = 0.0002
 
X = tf.placeholder(tf.float32, [None, 64, 64, 3])
Z = tf.placeholder(tf.float32, [None, NoiseSize])
IsTraining = tf.placeholder(tf.bool)

DiscGlobalStep = tf.Variable(0, trainable = False, name = "DiscGlobal")
GeneGlobalStep = tf.Variable(0, trainable = False, name = "GeneGlobal")
 
    
    
"""
Step 1. Generator에 노이즈를 입력하여 가짜 이미지를 생성
Step 2. Discriminator에 진짜 이미지 (MNIST)를 입력하여 Real 값을 추출
Step 3. Discriminator의 변수를 고정하고 (reuse), 가짜 이미지를 Discriminator에 넣어서 Gene 값을 추출

진짜 이미지와 가짜 이미지가 판별기를 지났을때 추출한 output으로 Discriminator의 손실함수를 정의한다.
Discriminator의 손실도는 이 둘을 더한 것으로서 수식적 형태로는 "log R + log (1-G)" 이다.
DiscReal은 1, DiscGene은 0이 되도록 신경망을 경쟁시킨다.
"""
Fake = Build_Generator(Z)
DiscReal = Build_Discriminator(X)
DiscGene = Build_Discriminator(Fake, True)
 
LossDiscReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DiscReal, labels=tf.ones_like(DiscReal)))
LossDiscGene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DiscGene, labels=tf.zeros_like(DiscGene)))

LossDisc = LossDiscReal + LossDiscGene
LossGene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DiscGene, labels=tf.ones_like(DiscGene)))



"""
tf.get_collection과 tf.GraphKeys.TRAINABLE_VARIABLES with "socpe 이름" 을 이용하여 독립으로 학습시킨 변수들의 묶음을 정의한다.
Discriminator의 변수와 Generator의 변수는 따로 학습시킨다.
tf.control_dependecies는 묶음 연산과 실행 순서를 정의하는 메서드이다.
UpdataOps를 먼저 실행하고 TrainDisc, TrainGene을 실행한다.
"""
DiscVars = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, scope = "DiscriminatorVal")
GeneVars = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, scope = "GeneratorVal")
UpdateOps = tf.get_collection (tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(UpdateOps):
    TrainDisc = tf.train.AdamOptimizer(LearningRate).minimize(LossDisc, var_list=DiscVars)
    TrainGene = tf.train.AdamOptimizer(LearningRate).minimize(LossGene, var_list=GeneVars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    # 1 epoch는 55000장의 MNIST 이미지들을 한 번씩 학습에 사용한 것
    for epoch in range(100000) :
        
        LossDiscVal, LossGeneVal = 0, 0

        # 55000장의 이미지를 한 번에 학습에 사용할 수는 없다. 그래서 BatchSize = 128 단위로 쪼개서 학습 TotalBatch = 55000/128이다.
        for i in range(8) :
            
            BatchImage = Build_Shuffle_BatchData (BatchSize, GetImage)
            Noise = Build_GetNoise(BatchSize, NoiseSize)

            _, LossDiscVal = sess.run([TrainDisc, LossDisc], feed_dict={X: BatchImage, Z: Noise, IsTraining: True})
            _, LossGeneVal = sess.run([TrainGene, LossGene], feed_dict={X: BatchImage, Z: Noise, IsTraining: True})
            """
            LossGeneVal 학습 시, feed_dict에 X : batch_xs를 넣어주어야 하는 이유를 잘 모르겠음
            """

        print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(LossDiscVal), 'G loss: {:.4}'.format(LossGeneVal))

        if epoch % 1 == 0 :
            
            Noise = Build_GetNoise(10, NoiseSize)
            Samples1 = sess.run(Fake, feed_dict = {Z: Noise, IsTraining: False})
            Samples2 = sess.run(Fake, feed_dict = {Z: Noise, IsTraining: True})
            
            fig, ax = plt.subplots(2, 5, figsize=(10, 5))

            for i in range(5) :
                
                ax[0][i].set_axis_off()
                ax[0][i].imshow(Samples1[i])
                ax[1][i].set_axis_off()
                ax[1][i].imshow(Samples2[i])

            plt.show ()
            plt.close(fig)
