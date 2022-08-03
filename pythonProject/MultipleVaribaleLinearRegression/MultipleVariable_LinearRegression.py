##################################################################
# [학습에 필요한 모듈 선언]
##################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 3차원 공간에서 그래프 출력
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

##################################################################
# [환경설정]
##################################################################
# 학습 데이터 수 선언
trainDataNumber = 200
# 모델 최적화를 위한 학습률 선언
learningRate = 0.0001
# 총 학습 횟수 선언
totalStep = 1001

##################################################################
# [빌드단계]
# Step 1) 학습 데이터 준비
##################################################################
# 항상 같은 난수를 생성하기 위하여 시드설정
np.random.seed(321)

# 학습 데이터 리스트 선언
x1TrainData = list()
x2TrainData = list()
yTrainData = list()

# 학습 데이터 생성
x1TrainData = np.random.normal(0.0, 1.0, size=trainDataNumber)
x2TrainData = np.random.normal(0.0, 1.0, size=trainDataNumber)

for i in range(0, trainDataNumber):
    # y데이터 생성
    x1 = x1TrainData[i]
    x2 = x2TrainData[i]
    y = 10 * x1 + 5.5 * x2 + 3 + np.random.normal(0.0, 3)
    yTrainData.append(y)


x1TrainData = [2.2, 3.1, 15.4, 7.6, 11, 9.2, 3.4, 13.7, 4.5, 10, 2, 1.5, 1, 3, 4, 5, 8.5, 12, 18, 2.5, 2, 1.5, 10, 5, 20,
              30, 7, 12, 15, 18, 2, 1.5, 5, 10, 8, 8.4, 22, 18, 4.5, 0.8, 15, 2, 1.5, 10, 9, 0.5, 8.2, 5, 2.5, 1, 20] # 거리

x2TrainData = [22, 31, 15.4, 7.6, 11, 9.2, 3.4, 13.7, 4.5, 10, 2, 1.5, 1, 3, 4, 5, 8.5, 12, 18, 2.5, 2, 1.5, 10, 5, 20,
              30, 7, 12, 15, 18, 2, 1.5, 5, 10, 8, 8.4, 22, 18, 4.5, 0.8, 15, 2, 1.5, 10, 9, 0.5, 8.2, 5, 2.5, 1, 20] # 시간

yTrainData = [4000, 5700, 27300, 9000, 18000, 7400, 4800, 24000, 5600, 17700, 4000, 3000, 2500, 7500, 9500, 11000,
              16000, 30000, 43000, 5000, 4000, 3000, 15600, 10000, 34000, 54000, 12300, 18000, 20000, 33000, 4000, 3800,
              9600, 21000, 16400, 17200, 49600, 34900, 8200, 3800, 31900, 4000, 3000, 18000, 16300, 2000, 15200, 9700,
              5200, 2100, 37000] # 요금

# 학습 데이터 확인
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1TrainData,
        x2TrainData,
        yTrainData,
        linestyle="none",
        marker="o",
        mfc="none",
        markeredgecolor="red")
plt.show()

##################################################################
# [빌드단계]
# Step 2) 모델 생성을 위한 변수 초기화
##################################################################
# Weight 변수 선언
W1 = tf.Variable(tf.random_uniform([1]))       # 0~1사이의 균등확률분포 값을 생성하여 변수에 저장
W2 = tf.Variable(tf.random_uniform([1]))
# Bias 변수 선언
b = tf.Variable(tf.random_uniform([1]))

# 학습 데이터 x1TrainData, x2TrainData가 들어갈 플레이스 홀더 선언
X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
# 학습 데이터 yTrainData가 들어갈 플레이스 홀더 선언
Y = tf.placeholder(tf.float32)

##################################################################
# [빌드단계]
# Step 3) 학습 모델 그래프 구성
##################################################################
# 3-1) 학습 데이터를 대표 하는 가설 그래프 선언
hypothesis = W1 * X1 + W2 * X2 + b

# 3-2) 비용함수(오차함수,손실함수) 선언
costFunction = tf.reduce_mean(tf.square(hypothesis - Y)) # (예상값 - 실제 값)을 제곱하고 모두 더한 평균

# 3-3) 비용함수의 값이 최소가 되도록 하는 최적화함수(GradientDescent) 선언
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
train = optimizer.minimize(costFunction)

##################################################################
# [실행단계]
# 학습 모델 그래프를 실행
##################################################################
# 실행을 위한 세션 선언
sess = tf.Session()
# 최적화 과정을 통하여 구해질 변수 W,b 초기화
sess.run(tf.global_variables_initializer())

# 학습 데이터와 학습 결과를 matplotlib를 이용하여 결과 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1TrainData,
        x2TrainData,
        yTrainData,
        linestyle="none",
        marker="o",
        mfc="none",
        markeredgecolor="red")

Xs = np.arange(min(x1TrainData), max(x1TrainData), 0.05)
Ys = np.arange(min(x2TrainData), max(x2TrainData), 0.05)
Xs, Ys = np.meshgrid(Xs, Ys)

print("--------------------------------------------------------------------------------")
print("Train(Optimization) Start")
# totalStep 횟수 만큼 학습
for step in range(totalStep):
    # X, Y에 학습데이터 입력하여 비용함수, W, b, train을 실행
    cost_val, W1_val, W2_val, b_val, _ = sess.run([costFunction, W1, W2, b, train],
                                                  feed_dict={X1: x1TrainData,
                                                             X2: x2TrainData,
                                                             Y: yTrainData})
    # 학습 50회 마다 중간 결과 출력
    if step % 10 == 0:
        print("Step : {}, cost : {}, W1 : {}, W2 : {}, b : {}".format(step,
                                                                      cost_val,
                                                                      W1_val,
                                                                      W2_val,
                                                                      b_val))
        # 학습 단계 중간결과 Fitting Surface 추가
        if step % 20 == 0:
            ax.plot_surface(Xs,
                            Ys,
                            W1_val * Xs + W2_val * Ys + b_val,
                            rstride=4,
                            cstride=4,
                            alpha=0.2,
                            cmap=cm.jet)
print("Train Finished")
print("--------------------------------------------------------------------------------")
# 결과 확인 그래프
plt.show()

#세션종료
sess.close()