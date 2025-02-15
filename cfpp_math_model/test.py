import math

# 计算两点之间的距离
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

X = [119.775858, 36.349404]
Y = [119.773650, 36.348890]
print(eucliDist(X,Y))
