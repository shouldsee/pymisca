# from inspect import signature
from functools import wraps
import decorator
# inspect.get
def dec(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def dec2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def dec3(func):
    @decorator.decorator
    def wrapper(f,*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def foo(a, b=1):
    pass

signature = inspect.getargspec
print(signature(dec(foo)))
print(signature(dec2(foo)))
print(signature(dec3(foo)(foo)))