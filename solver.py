from sympy.parsing.sympy_parser import parse_expr
from iic import MathSymbolClassifier
from contour import Contour
from enums import BarType
import cv2 as cv


class Solver:
    def __init__(self):
        self.cl = MathSymbolClassifier()

    def get_equation(self, contourList, frame):
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
        equation = []

        for el in temp:
            if isinstance(el, str):
                equation.append(el)
            elif el.barType == BarType.FRACTION_HOR or el.barType == BarType.FRACTION_VERT:
                cv.drawContours(
                    frame, [el.contour], -1, (0, 255, 0), 2)
                equation.append('/')
            elif el.barType == BarType.EQUAL:
                cv.drawContours(
                    frame, [el.contour], -1, (0, 0, 255), 2)
                equation.append('=')
            elif el.barType == BarType.MINUS:
                cv.drawContours(
                    frame, [el.contour], -1, (255, 0, 0), 2)
                equation.append('-')
            elif el.barType == BarType.COMMA:
                cv.drawContours(
                    frame, [el.contour], -1, (255, 0, 255), 2)
                equation.append('.')
            elif el.barType == BarType.MULTIPLY:
                cv.drawContours(
                    frame, [el.contour], -1, (0, 255, 255), 2)
                equation.append('*')
            else:
                cv.drawContours(
                    frame, [el.contour], -1, (255, 255, 0), 2)
                symbol = self.cl.classify(
                    [el.get_subimage_for_classifier()])[0]
                equation.append(symbol)

        equationString = "".join(equation)
        print(equationString)
        return equationString

    def solve(self, contourList, frame):
        equationString = self.get_equation(contourList, frame)
        # expr = parse_expr(equationString)
        # solution = expr.evalf(4)
        # return solution
        # print(expr)


if __name__ == '__main__':
    eq = '(1/2 + 4 * 3.5) / 2 + 2.00001'
    eq = [[['1', '/', '2'], '+', ['4', '*', '3', '.', '5']],
          '/', '2', '+', '2', '.', '00001']
    print(Solver().solve(eq))
