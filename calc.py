import os
# import matplotlib.pyplot as plt
# from sympy import *
import numpy as np
import math

# eps = 60


class calcul:
    eps = 60
    eps1 = math.tan(eps)

    R2 = 289
    R1 = 287
    z1 = 1
    z2 = 1
    r1 = 1
    T1 = 300
    k1 = 1.4
    k2 = 1.33
    theta2 = np.array([1, 2, 3, 4, 5, 6])
    allM1 = np.array([0.1, 0.4, 0.8])
    a1 = np.sqrt(k1 * R1 * T1)
    delta = 0.1
    ms = []
    allr = []
    r2s = np.array([0.6, 0.7, 0.8])
    rs = []
    Pi0 = np.array([1.5, 3, 6])




    @staticmethod
    def peripheral(R1, R2, delta, eps1, r2s, allM1, a1, theta2, ms, k1, T1):    #периферийный вихрь

        ''' нет 3х эпюр'''
        ''' эпюры '''
        schet = 0
        wzs = []
        mzs = []

        mfs = []
        wfs = []

        Ms = []
        ws = []
        '''  ___  '''

        sqreps = (1 + eps1 ** 2)

        for i, r2 in enumerate(r2s):

            rm = ((1 + r2) / 2) + delta

            rs = np.arange(r2, 1, 0.05)

            r0 = 2 * rm - r2

            for j, theta in enumerate(theta2):
                m = 1 + ((math.log10(R1 / R2)) - (math.log10(theta))) * (1 / (math.log10(r2)))
                ms.append(m)

            for k, r in enumerate(rs):


                for l, M1 in enumerate(allM1):
                    mz1 = M1 / (sqreps ** 0.5)

                    w1 = M1 * a1

                    wz1 = mz1 * a1

                    mf1 = mz1 * eps1

                    wf1 = mf1 * a1

                    a = mz1 / ((1 - r2) * (r2 - 2 * rm + 1))

                    b = -2 * a * rm

                    c = mz1 - a - b

                    n = ((a * (rm - 1)) ** 2) / mz1

                    _mzm = 1 + n

                    mzm = mz1 * _mzm

                    mz = (a * r ** 2) + b * r + c
                    mzs.append(mz)

                    for p, m in enumerate(ms):

                        A = ((k1 - 1)/(2 * m)) * ((M1 ** 2)/(1 + sqreps))

                        func1 = (1 - A * ((1 / (r ** (2 * m))) - 1))
                        func2 = np.sqrt(k1 * R1 * T1 * func1)

                        mf = M1 / ((sqreps ** 0.5 ) * (func1 ** 0.5) * (r ** (2 * m)))
                        mfs.append(mf)

                        M = np.sqrt((mf ** 2) + (mz ** 2))
                        Ms.append(M)

                        wf = (M1 / ((sqreps ** 0.5) * (r ** m))) * func2
                        wfs.append(wf)

                        wz = mz * func2
                        schet += 1
                        wzs.append(wz)

                        print(schet)


    @staticmethod
    def exial(R1, R2, delta, eps1, r2s, allM1, a1, theta2, ms, k1, k2, T1, z1, z2, Pi0):        #приосевой вихрь


        sqreps = (1 + eps1 ** 2)
        for i, r2 in enumerate(r2s):

            rs = np.arange(0, r2, 0.05)

            for o, pi0 in enumerate(Pi0):

                for j, theta in enumerate(theta2):
                    m = 1 + ((math.log10(R1 / R2)) - (math.log10(theta))) * (1 / (math.log10(r2)))
                    ms.append(m)

                    for l, M1 in enumerate(allM1):

                        mz1 = M1 / (sqreps ** 0.5)

                        w1 = M1 * a1

                        wz1 = mz1 * a1

                        mf1 = mz1 * eps1

                        for p, m in enumerate(ms):

                            A = ((k1 - 1) / (2 * m)) * ((M1 ** 2) / (1 + sqreps))

                            fi2 = 1 - A * ((1 / (r2 ** (2 * m))) - 1)

                            gamma = ((k1 * (k2 - 1))/(2 * k2)) * ((z1 * R1)/(z2 * R2)) * ((M1 ** 2)/((r2 ** (2 * m)) *
                                                                                                theta2 * fi2 * sqreps *
                                    (1 - ((1/pi0) ** ((k2 - 1)/k2)) * (1/(fi2 ** ((k1 * (k2 - 1))/(k2 * (k1 - 1))))))))

                            B = ((z1 * R1)/(z2 * R2)) * (((k2 - 1) * k1 * (M1 ** 2))/(2 * gamma * k2 * sqreps *
                                                                                      (r2 ** (2 * m)) * theta2 * fi2))

                            for k, r in enumerate(rs):

                                Mf = mf1 * (1/(r2 ** m)) * ((r/r2) ** gamma) * (((k1 * R1)/(k2 * R2)) ** 0.5) * (1/((fi2 ** 0.5) * ((1 - B * (1 - ((r/r2) ** (2 * gamma)))) ** 0.5)))








    @staticmethod
    def run():
        calcul.peripheral(calcul.R1, calcul.R2, calcul.delta,
                          calcul.eps1, calcul.r2s, calcul.allM1,
                          calcul.a1, calcul.theta2, calcul.ms,
                          calcul.k1, calcul.T1)

if __name__ == '__main__':
    calcul.run()
