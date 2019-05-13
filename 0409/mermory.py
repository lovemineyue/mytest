import sys
def get_size(obj, seen=None):
    # From
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
# Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
      size += sum([get_size(v, seen) for v in obj.values()])
      size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
      size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
      size += sum([get_size(i, seen) for i in obj])
    return size


# class DataItem(object):
#     def __init__(self, name, age, address):
#         self.name = name
#         self.age = age
#         self.address = address

class DataItem(object):
    __slots__ = ['name', 'age', 'address']
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address


d1 = DataItem("Alex", 42, "-")
print ("sys.getsizeof(d1):", sys.getsizeof(d1))


d1 = DataItem("Alex", 42, "-")
print ("get_size(d1):", get_size(d1))
d2 = DataItem("Boris", 24, "In the middle of nowhere")
print ("get_size(d2):", get_size(d2))
