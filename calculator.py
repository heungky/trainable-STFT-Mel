# calculator.py
import argparse
import Operator.operator as Op

# create the parser
calculator = argparse.ArgumentParser(description='Calculator')


# add the arguments (must)
# metavar: provides a diff name for optional argument in help messages 
# indicate how many variable gonna accept

calculator.add_argument('Number1',metavar='1st number', type=float, 
                        help='the number that you want to process')

calculator.add_argument('Number2',metavar='2nd number', type=float, 
                        help='the number that you want to process')


calculator.add_argument('--mode', type=str, help='Type of function you want to use in calculator.Can choose sum/ multiply/ subtract/ division')


# class Operator():
#     def summation(self, a,b):
#         return a+b

#     def multiply(self, a,b):
#         return a*b

#     def subtraction(self, a,b):
#         return a-b

#     def division(self, a,b):
#         return a/b    





# execute the parse_args() method
# take argument become variable 
number1 = calculator.parse_args().Number1
number2 = calculator.parse_args().Number2

function = calculator.parse_args().mode

operator = getattr(Op, function)()

print(operator(number1, number2))

# if function=='sum':
#     print(summation(number1, number2))
# elif function=='multiply':
#     print(multiply(number1, number2))
# elif function=='subtract':
#     print(subtraction(number1, number2))
# elif function=='division':
#     print(division(number1, number2))
# else:
#     print(f'Please try again, type sum/ multiply/ substract/ division')
