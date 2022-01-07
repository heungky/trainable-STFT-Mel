# calculator.py
import argparse


# create the parser
calculator = argparse.ArgumentParser(description='Calculator')



# add the arguments (must)
# metavar: provides a diff name for optional argument in help messages 
# indicate how many variable gonna accept

calculator.add_argument('Number1',metavar='1st number', type=float, 
                        help='the number that you want to add')

calculator.add_argument('Number2',metavar='2nd number', type=float, 
                        help='the number that you want to add')


calculator.add_argument('--mode', type=str, help='type of function you want to use in calculator')


def summation(a,b):
    return a+b

def multiply(a,b):
    return a*b

# execute the parse_args() method
# take argument become variable 
number1 = calculator.parse_args().Number1
number2 = calculator.parse_args().Number2
mode = calculator.parse_args().mode

if mode=='summation':
    print(summation(number1, number2))
elif mode=='multiply':
    print(multiply(number1, number2))
else:
    pass
