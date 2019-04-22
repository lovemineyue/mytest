def log(func):
    print('aaa')
    def haha(*args, **kwargs):
        print("cccc")
        a = func(*args, **kwargs)
        print('ddddd')
        return a

    print('bbbb')
    return haha


@log
def test():
    print('wo in China')

test()


# def log(text):
#     def decorator(func):
#         def wrapper(*args, **kw):
#             print('%s %s():' % (text, func.__name__))
#             return func(*args, **kw)
#         return wrapper
#     return decorator
#
# @log('execute')
# def now():
#     print('2015-3-25')
#
# now()
