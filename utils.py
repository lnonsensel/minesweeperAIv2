import typing as tp
import time
import inspect
import numpy as np



USE_LOGGER = False
total = {}
def log(func: tp.Callable):
    def wrapper(*args, **kwargs):
        if USE_LOGGER:
            print(f'{func.__name__} {inspect.getfile(func)} function running')
            start_time = time.time()
        result = func(*args, **kwargs)
        if USE_LOGGER:
            func_runtime = round(time.time() - start_time, 2)
            if total.get(func.__name__) is None:
                total[func.__name__] = func_runtime
                print(f'{func.__name__} {inspect.getfile(func)} function was running for {func_runtime}, total {total[func.__name__]}')
            else:
                total[func.__name__] += func_runtime
        return result
    return wrapper
