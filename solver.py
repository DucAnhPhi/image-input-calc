from sympy.parsing.sympy_parser import parse_expr
from iic import MathSymbolClassifier
from contour import Contour


class Solver:
    def __init__(self, contourList):
        def add_brackets(cnt):
            if not isinstance(cnt, list):
                return cnt
            else:
                return ["(", *[add_brackets(c) for c in cnt], ")"]

        equation = [add_brackets(el) for el in contourList]

        def flatten(nestedList):
            for el in nestedList:
                if isinstance(el, list):
                    yield from flatten(el)
                else:
                    yield el

        equation = flatten(equation)

        equation = [
            '/' if isinstance(el, Contour) and el.isFractionBar else el for el in equation]

        cl = MathSymbolClassifier()
        equation = [cl.classify(
            [el.get_subimage()])[0] if isinstance(el, Contour) else el for el in equation]

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
