list = [2,7,11,15]

# def twoSum(sums,target):

# print(dict(list))

a = [{i,x} for i,x in enumerate(list)]
target = 9
def sum(target, *sums):
    for i ,x in sums:
        for z, y in sums:
            if x + y == target:
                return [i, z]

s = sum(target, *a)

print(s)

from mxnet import nd

a = nd.array([1,2])

print(a)
