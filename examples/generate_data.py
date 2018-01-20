"""Module to generate a data set of binary and normal decimals

For example [(0, 0), (1, 1), (10, 2), (11, 3), (100, 4), (101, 5), (110, 6), (111, 7), (1000, 8), (1001, 9)] for limit = 10
""" 

import numpy as np
import math

def generate_data(sign, n_examples=10000, limit=100):
    """Generate data based on the sign"""
    if sign == "+":
        return generate_sum_data(n_examples, limit)
    elif sign == "-":
        return generate_difference_data(n_examples, limit)
    elif sign == "*":
        return generate_product_data(n_examples, limit)
    elif sign == "/":
        return generate_fraction_data(n_examples, limit)
    else:
        return NotImplementedError

def generate_binary_data(limit=10000, write=False):
    """Function that generates (binary,decimal) data till the limit"""
    DATA = []
    for i in range(0,limit):
        DATA.append((_decimal_to_binary(i),i))
    if write: 
        _write_to_file(DATA)
    return DATA

def generate_sum_data(n_examples, limit):
    """Generate 'n_examples' pairs of numbers and their sum,
    limit gives max number value"""
    X, y = zip(*[((n1, n2), n1+n2) for n1, n2 in [tuple(np.random.randint(0, limit, 2)) for _ in range(n_examples)]]) 
    return X, y

def generate_difference_data(n_examples, limit):
    """Generate 'n_examples' pairs of numbers and their difference,
    limit gives max number value"""
    X, y = zip(*[((n3, n1), n2) for (n1, n2), n3 in zip(*generate_sum_data(n_examples, limit))])
    return X, y

def generate_product_data(n_examples, limit):
    """Generate 'n_examples' pairs of numbers and their product,
    limit gives max number value"""
    X, y = zip(*[((n1, n2), n1*n2) for n1, n2 in [tuple(np.random.randint(0, limit, 2)) for _ in range(n_examples)]]) 
    return X, y

def generate_fraction_data(n_examples, limit):
    """Generate 'n_examples' pairs of numbers and their fraction,
    limit gives max number value"""
    X, y = zip(*[((n3, n1), (n2 if n3+n1 != 0 else 0)) for (n1, n2), n3 in zip(*generate_product_data(n_examples, limit))])
    return X, y
 
def _decimal_to_binary(decimal):
    """Convert decimal to binary"""
    return int("{0:b}".format(decimal))

def _write_to_file(data, filename="data.txt"):
    """Writes given data to file"""
    FILE = open(filename, "w")
    for row in data:
        FILE.write("{},{}\n".format(row[0],row[1]))
    print("done")