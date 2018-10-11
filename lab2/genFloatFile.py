#!/usr/bin/python3
import sys
import random
file_name="float_data"
f=open(file_name,mode="w")
num_floats=sys.argv[1]
#generation of floats
random.seed()
for i in range(int(num_floats)):
    n=random.uniform(0,100)
    f.write("{random_float:f}\n".format(random_float=n))
