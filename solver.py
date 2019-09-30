from sympy.parsing.sympy_parser import parse_expr
from classify import MathSymbolClassifier
from contour import Contour
from enums import MathSign
from enums import Position
import cv2 as cv


class Solver:
    def __init__(self):
        models = {'model-digits-plus.ckpt':11}
        self.cl = MathSymbolClassifier(num_classes=11, models=models)

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
            elif el.mathSign == MathSign.FRACTION_HOR or el.mathSign == MathSign.FRACTION_VERT:
                cv.drawContours(
                    frame, [el.contour], -1, (0, 255, 0), 2)
                equation.append('/')
            elif el.mathSign == MathSign.EQUAL:
                cv.drawContours(
                    frame, [el.contour], -1, (0, 0, 255), 2)
                equation.append('=')
            elif el.mathSign == MathSign.MINUS:
                cv.drawContours(
                    frame, [el.contour], -1, (255, 0, 0), 2)
                equation.append('-')
            elif el.mathSign == MathSign.COMMA:
                cv.drawContours(
                    frame, [el.contour], -1, (255, 0, 255), 2)
                equation.append('.')
            elif el.mathSign == MathSign.MULTIPLY:
                cv.drawContours(
                    frame, [el.contour], -1, (0, 255, 255), 2)
                equation.append('*')
            else:
                cv.drawContours(
                    frame, [el.contour, *el.holes], -1, (255, 255, 0), 2)
                symbol = self.cl.classify(
                    [el.get_subimage_for_classifier()])[0]
                # if symbol.isdigit() and isExponent:
                #     equation.append('^')
                equation.append(symbol)

        equationString = "".join(equation)
        return equationString

    def solve(self, contourList, frame):
        equationString = self.get_equation(contourList, frame)
        equationString = equationString.replace("=", "")
        try:
            expr = parse_expr(equationString)
            solution = expr.evalf(4)
            print(equationString + " = " + str(solution))
        except:
            print(equationString+" = " + "?")
