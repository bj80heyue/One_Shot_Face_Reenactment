import multiprocessing
import time
from tqdm import *

# list = [1, 2, 3, 4]
def func(i):
    msg = "hello %d" % (list[i])
    print ("msg:", msg)
    time.sleep(3)
    print("end")


list = [1, 2, 3, 4]
data = []
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 4)
    # list = [1, 2, 3, 4]
    for i in tqdm(range(4)):
        data.append(i)
        pool.map(func, data)
        # pool.apply_async(func, (i, list, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

    print("Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~")
    pool.close()
    pool.join()
    print("Sub-process(es) done.")