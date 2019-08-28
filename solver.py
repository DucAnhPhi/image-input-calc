from sympy.parsing.sympy_parser import parse_expr


class Solver:
    def solve(self, exprStr):
        expr = parse_expr(exprStr)
        solution = expr.evalf(4)
        return solution


if __name__ == '__main__':
    eq = '(1/2 + 4 * 3.5) / 2 + 2.00001'
    print(Solver().solve(eq))
