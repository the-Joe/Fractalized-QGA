import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#a = tf.constant(1.)
#b = tf.constant(5.)
#c = tf.constant(2.)
#d = tf.constant(5.)
#e = tf.constant(6.)

#result = ((a+b+c)*d)/6
#x = a + b + c
#y = x*d
#result = y/6

sess = tf.compat.v1.Session()
float = tf.compat.v1.float32()

#r = sess.run(result)
#print(sess.run(x))
#print(sess.run(y))
#print (r)

a = tf.constant(float)
b = tf.constant(float)
c = tf.constant(float)
d = tf.constant(float)
e = tf.constant(float)

result = ((a+b+c)*d)/6

print(sess.run(result, feed_dict={a:1,b:5,c:2,d:5,e:6}))