# # array test
# array = [None]*10
# print('array =', array)

# array[3] = 'd'
# array[7] = 'h'
# array[1] = 'b'
# array[0] = 'a'
# array[2] = 'c'
# array[4] = 'e'
# array[6] = 'g'
# array[5] = 'f'
# array[9] = 'j'
# array[8] = 'i'

# print('array =', array)

from random import seed
from random import randint


print(randint(1, 10))
# for _ in range(5):
# 	print(random())
# # seed the generator to get the same sequence
# print('Reseeded')
# seed(5)
# for _ in range(5):
# 	print(random())