import time


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())



if __name__ == '__main__':
    print(get_current_time())
    time.sleep(5)
    print(get_current_time())