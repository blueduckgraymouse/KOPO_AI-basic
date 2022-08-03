# import tensorflow as tf
#
# x = tf.constant(3)
# print(x)    # -> Tensor("Const:0", shape=(), dtype=int32)
#
#
# xx = tf.constant(3)
# sess = tf.Session()
# result = sess.run(xx)
# print(result)   # -> 3
#
# print(x)    # -> Tensor("Const:0", shape=(), dtype=int32)



###



# import tensorflow as tf
# var_1 = tf.Variable(3)
# var_2 = tf.Variable(10)
# result_sum = var_1 + var_2
# sess = tf.Session()
# print(sess.run(result_sum))       #초기화 함수가 없으므로 에러가 발생


import tensorflow as tf
var_1 = tf.Variable(3)
var_2 = tf.Variable(10)
result_sum = var_1 + var_2
init = tf.global_variables_initializer()     # 초기화 함수 추가
sess = tf.Session()
sess.run(init)                 	      # 초기화된 결과를 세센에 전달
print(sess.run(result_sum))



