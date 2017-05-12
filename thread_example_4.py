#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:30:26 2017

@author: song-46
"""

import threading
from queue import Queue

def func(l, q):
    for i in range(len(l)):
        l[i] = l[i]**2
    q.put(l)
    
def myThread(data):
    q = Queue()
    threads = []
    for i in range(len(data)):
        t = threading.Thread(target=func, args=(data[i], q), name="T%d"%i)
        threads.append(t)
        
    for thread in threads:
        thread.start()
        thread.join()
        print(thread.getName())
        
    results = []
    for i in range(len(data)):
        results.append(q.get())
    print(results)
        
if __name__ == "__main__":
    data = [[1, 2, 3], [2, 3, 4]]
    myThread(data)