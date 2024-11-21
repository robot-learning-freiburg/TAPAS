import multiprocessing


def mp_wrapper(func):
    def wrapper(*args, **kwargs):
        launch_in_mp(func, *args, **kwargs)

    return wrapper


def launch_in_mp(func, *args, **kwargs):
    multiprocessing.Process(target=func, args=args, kwargs=kwargs).start()
