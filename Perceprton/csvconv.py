import csv
import random
#convert csv to float list
def tofloat(csvfile):
    floatdata=list()
    with open(csvfile) as f:
        dataset = f.readlines()
        for instance in dataset:
            row = [float(attribute) for attribute in instance.split(',')]
            floatdata.append(row)
    return floatdata