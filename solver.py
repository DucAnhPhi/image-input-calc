from sympy.parsing.sympy_parser import parse_expr
from iic import MathSymbolClassifier
from contour import Contour
from enums import BarType


class Solver:
    def __init__(self, contourList):
        def add_brackets(cnt):
            if not isinstance(cnt, list):
                return cnt
            else:
                return ["(", *[add_brackets(c) for c in cnt], ")"]

        temp = [add_brackets(el) for el in contourList]

        def flatten(nestedList):
            for el in nestedList:
                if isinstance(el, list):
                    yield from flatten(el)
                else:
                    yield el

        temp = flatten(temp)
        cl = MathSymbolClassifier()
        equation = []

        for el in temp:
            if isinstance(el, str):
                equation.append(el)
            elif el.barType == BarType.FRACTION:
                equation.append('/')
            elif el.barType == BarType.EQUAL:
                equation.append('=')
            elif el.barType == BarType.MINUS:
                equation.append('-')
            else:
                symbol = cl.classify([el.get_subimage_for_classifier()])[0]
                equation.append(symbol)

        self.equation = "".join(equation)
        print(self.equation)

    def solve(self):
        expr = parse_expr(self.equation)
        solution = expr.evalf(4)
        return solution


if __name__ == '__main__':
    eq = '(1/2 + 4 * 3.5) / 2 + 2.00001'
    eq = [[['1', '/', '2'], '+', ['4', '*', '3', '.', '5']],
          '/', '2', '+', '2', '.', '00001']
    print(Solver(eq).solve())
