import math
from sympy import *
import numpy as np
from fractions import Fraction


class Calc:
    Eps = np.array([Fraction(0.577), Fraction(1), Fraction(1.732), Fraction(3.732), Fraction(11.43)])
    R2 = 288.3
    R1 = 287
    z1 = 1
    z2 = 1

    k1 = 1.4
    k2 = 1.33

    @staticmethod
    def main_calc(piks, pi0, theta2, r_vnesh, r_vnutr, r2, eps, rm, r):
        ''' not finished '''

        # b = (2 * rm) / ((1 - r2) * ((2 * rm) - r2 - 1))
        b = (2 * rm) / ((r2 ** 2) - (2 * rm * r2) + 1 + 2 * rm)

        a = -1 * (b / (2 * rm))

        c = 1 - a - b

        m = (np.log(r2 * (Calc.R1 / Calc.R2) * (1 / theta2)))

        M1 = np.sqrt(2 * (((piks ** (1 / (Calc.k1 - 1) / Calc.k1)) - 1) / (Calc.k1 - 1)))

        A = ((Calc.k1 - 1) / (2 * m)) * (M1 ** 2) / (1 + (1 / eps ** 2))

        mf1 = M1 / ((1 + (1 / (eps ** 2))) ** 0.5)

        mz1 = (((M1 ** 2) / (1 + eps ** 2)) ** (1 / 2))

        Fi2 = 1 - A * (1 / ((r2 ** (2 * m)) - 1))

        gamma = ((Calc.k1 * (Calc.k2 - 1) * Calc.z1 * Calc.R1) / (2 * Calc.k2 * Calc.z2 * Calc.R2)) * \
                ((M1 ** 2) / ((1 + (1 / eps ** 2)) * (r2 ** (2 * m)) * theta2 * Fi2) *
                 (1 - ((1 / pi0) ** ((Calc.k2 - 1) / Calc.k2))) * (
                 1 / (Fi2 ** ((Calc.k1 * (Calc.k2 - 1)) / (Calc.k2 * (Calc.k1 - 1))))))

        B = (Calc.k1 * (Calc.k2 - 1) * (M1 ** 2) * Calc.z1 * Calc.R1) / \
            (2 * Calc.k2 * (1 + (1 / (eps ** 2))) * (r2 ** (2 * m)) * theta2 * Fi2 * Calc.z2 * Calc.R2)

        FiSt3 = (1 - ((0.2 * M1 ** 2) / ((1 + (1 / (eps ** 2))) * m) * ((1 / (r ** (2 * m))) - 1)) ** 3)

        _Mzm = a * (rm ** 2) + b * rm + c

        _Mz = a * (r ** 2) + b * r + c

        if r_vnesh is not None:
            mf_vnesh = M1 / \
                       (
                       (1 + 1 / eps ** 2) ** 0.5 * (1 - A * (1 / (r_vnesh ** (2 * m)) - 1)) ** 0.5 * r_vnesh ** (2 * m))

            x = symbols('x')

            f = FiSt3 * _Mz * r

            integr = integrate(f, (x, r2, 1))

            Fc = integr / (0.5 * (((eps ** 2) + 1) ** 0.5))

            print(Fc)

            return mf_vnesh

        if r_vnutr is not None:
            mf_vnutr = (1 / r2 ** m) * ((r_vnutr / r2) ** 2) * (((Calc.k1 * Calc.R1) / (Calc.k2 * Calc.R2)) ** 0.5) * \
                       (M1 / (((1 + (1 / eps ** 2)) ** 0.5) * ((1 - B * (1 - ((r_vnutr / r2) ** (2 * gamma)))) ** 0.5)))
            return mf_vnutr
