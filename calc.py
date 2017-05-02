import os
# import matplotlib.pyplot as plt
# from sympy import *
import numpy as np
import math
from multiprocessing import Pool



# eps = 60


class calcul:
    eps = 60
    eps1 = math.tan(eps)


    R1 = 287
    R2 = 289
    R3 = 283
    Re = 283 #????????????????
    z1 = 1
    z2 = 1
    z3 = 1
    ze = 1 #""" ?????????
    r1 = 1
    T1 = 300
    k1 = 1.4
    k2 = 1.33
    k3 = 1.33
    ke = 1.33
    theta2 = np.array([1, 2, 3, 4, 5, 6])
    allM1 = np.array([0.1, 0.4, 0.8])
    a1 = np.sqrt(k1 * R1 * T1)
    delta = 0.1
    ms = []
    allr = []
    r2s = np.array([0.6, 0.7, 0.8])
    rs = []
    Pi0 = np.array([1.5, 3, 6])
    Ksig = 1




    @staticmethod
    def peripheral(R1, R2, delta, eps1, r2s, allM1, a1, theta2, ms, k1, T1):    #периферийный вихрь

        ''' нет 3х эпюр'''
        ''' эпюры '''
        schet = 0

        wzsPer = []
        mzsPer = []

        mfsPer = []
        wfsPer = []

        MsPer = []
        wsPer = []
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
                    mzsPer.append(mz)

                    for p, m in enumerate(ms):

                        A = ((k1 - 1)/(2 * m)) * ((M1 ** 2)/(1 + sqreps))

                        func1 = (1 - A * ((1 / (r ** (2 * m))) - 1))
                        func2 = np.sqrt(k1 * R1 * T1 * func1)

                        mf = M1 / ((sqreps ** 0.5 ) * (func1 ** 0.5) * (r ** (2 * m)))
                        mfsPer.append(mf)

                        M = np.sqrt((mf ** 2) + (mz ** 2))
                        MsPer.append(M)

                        wf = (M1 / ((sqreps ** 0.5) * (r ** m))) * func2
                        wfsPer.append(wf)

                        wz = mz * func2
                        schet += 1
                        wzsPer.append(wz)

                        # print(schet)


    @staticmethod
    def axial(R1, R2, R3, Re, eps1, r2s, allM1, a1, theta2, ms, k1, k2, k3, ke, z1, z2, z3, ze, Pi0, Ksig):
        #приосевой вихрь

        mzs3zone = []
        mzs4zone = []
        mfsAx = []
        wfsAx = []
        Re = []
        r3zones = []
        r4zones = []

        sqreps = (1 + eps1 ** 2)

        for i, r2 in enumerate(r2s):

            rs = np.arange(0, r2, 0.05)

            rg = r2

            for o, pi0 in enumerate(Pi0):

                for j, theta in enumerate(theta2):

                    m = 1 + ((math.log10(R1 / R2)) - (math.log10(theta))) * (1 / (math.log10(r2)))

                    ms.append(m)

                    for l, M1 in enumerate(allM1):

                        mz1 = M1/(sqreps ** 0.5)

                        w1 = M1 * a1

                        wz1 = mz1 * a1

                        mf1 = mz1 * eps1

                        Piks = 1/(1 - (((M1 ** 2) * k1)/2))

                        Pie = Piks

                        for p, m in enumerate(ms):

                            A = ((k1 - 1) / (2 * m)) * ((M1 ** 2) / (1 + sqreps))

                            fi2 = 1 - A * ((1 / (r2 ** (2 * m))) - 1)

                            gamma = ((k1 * (k2 - 1))/(2 * k2)) * ((z1 * R1)/(z2 * R2)) * \
                                    ((M1 ** 2)/((r2 ** (2 * m)) * theta2 * fi2 * sqreps *
                                                (1 - ((1/pi0) ** ((k2 - 1)/k2)) *
                                                 (1/(fi2 ** ((k1 * (k2 - 1))/(k2 * (k1 - 1))))))))

                            B = ((z1 * R1)/(z2 * R2)) * (((k2 - 1) * k1 * (M1 ** 2))/
                                                         (2 * gamma * k2 * sqreps * (r2 ** (2 * m)) * theta2 * fi2))

                            C = ((k1 * (k3 - 1) * (M1 ** 2))/(2 * k3 * gamma * sqreps * fi2 * (r2 ** (2 * m))))\
                                * ((z1 * R1)/(z3 * R3))

                            re = rg * ((1 - (1/C) *
                                        (1 - ((1/(Pie * (fi2 ** (k1/(k1 - 1))))) ** ((k3 - 1)/k3)))) ** 1/(2 * gamma))

                            Re.append(re)


                            for h, re in enumerate(Re):

                                r3zone = np.arange(re, rg, 0.05)

                                # r3zones.append(r3zone)

                                r4zone = np.arange(0, re, 0.05)

                                # r4zones.append(r4zone)

                                for u, r3zone in enumerate(r3zones):

                                    ksi3zone = 1 - C * (1 - ((r3zone/rg) ** (2 * gamma)))

                                    mz3zone = np.sqrt((2/k3) * (1 - (1/(Piks * (fi2 ** (k1/(k1 - 1))) *
                                                            (Ksig ** (k2/(k2 - 1))) * (ksi3zone ** (k3/(k3 - 1)))))))

                                    mzs3zone.append(mz3zone)

                                for y, r4zone in enumerate(r4zones):

                                    ksi4zone = 1 - C * (1 - ((r4zone/re) ** (2 * gamma))) # Ksi E

                                    D = ((z1 * R1)/(ze * Re)) * ((re/r2) ** (2 * gamma)) *\
                                                                                        ((k1 * (ke - 1) * (M1 ** 2))/
                                         (2 * ke * gamma * sqreps * fi2 * Ksig * ksi4zone * (r2 ** (2 * m))))

                                    U = 1 - D * (1 - ((r4zone/re) ** (2 * gamma)))

                                    mz4zone = np.sqrt((2/k3) * ((1/(Piks * (fi2 ** (k1/(k1 - 1))) *
                                                                    (Ksig ** (k2/(k2 - 1))) *
                                                                    (ksi4zone ** (k3/(k3 - 1))) *
                                                                    (U ** (ke/(ke - 1))))) - 1))

                                    mzs4zone.append(mz4zone)

                                for k, r in enumerate(rs):

                                    MfAx = mf1 * (1/(r2 ** m)) * ((r/r2) ** gamma) * \
                                           (((k1 * R1)/(k2 * R2)) ** 0.5) * \
                                           (1/((fi2 ** 0.5) * ((1 - B * (1 - ((r/r2) ** (2 * gamma)))) ** 0.5)))

                                    mfsAx.append(MfAx)

                                    WfAx = a1 * MfAx

                                    wfsAx.append(WfAx)

    # @staticmethod
    # def run():
    #     calcul.peripheral(calcul.R1, calcul.R2, calcul.delta,
    #                       calcul.eps1, calcul.r2s, calcul.allM1,
    #                       calcul.a1, calcul.theta2, calcul.ms,
    #                       calcul.k1, calcul.T1)

if __name__ == '__main__':

    periph = calcul.peripheral(calcul.R1, calcul.R2, calcul.delta,
                          calcul.eps1, calcul.r2s, calcul.allM1,
                          calcul.a1, calcul.theta2, calcul.ms,
                          calcul.k1, calcul.T1)

    axial = calcul.axial(calcul.R1, calcul.R2, calcul.R3, calcul.Re, calcul.eps1, calcul.r2s,
                         calcul.allM1, calcul.a1, calcul.theta2, calcul.ms, calcul.k1, calcul.k2,
                         calcul.k3, calcul.ke, calcul.z1, calcul.z2, calcul.z3, calcul.ze, calcul.Pi0, calcul.Ksig)

    # pool = Pool()
    # pool.map(periph, axial)
    # pool.close()
    # pool.join()


    # calcul.run()
