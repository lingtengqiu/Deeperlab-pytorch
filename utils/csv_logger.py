#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2019-02-21 18:50
# * Last modified : 2019-02-21 18:50
# * Filename      : csv_logger.py
# * Description   : 
# **********************************************************
import numpy as np
import pandas as pd 
class csv_logger(object):
    def __init__(self,root):
        self.root = root
    def write(self,data,*columns):
        df = pd.DataFrame(data = data,columns = columns)
        print(self.root)
        df.to_csv(self.root,index = False,header = True)
if __name__ == "__main__":
    
    name=["xixi","haha"]
    val =[2,3]
    g = [1,4]
    print(zip(name,val,g))
    logger = csv_logger("./test.csv")
    logger.write(zip(name,val,g),*["name","val","g"])
    


    
