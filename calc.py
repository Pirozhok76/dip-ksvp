import os
# import matplotlib.pyplot as plt
# from sympy import *
import numpy as np
import math
from multiprocessing import Pool



# eps = 60


class calcul:


    epsi = np.array([60, 75, 85])

    # eps1 = math.tan(eps)

    # sqreps = 1 + eps ** 2

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
    # r2s = np.array([0.6, 0.7, 0.8])
    rs = []
    Pi0 = np.array([1.5, 3, 6])
    Ksig = 1




    @staticmethod
    def peripheral(eps, r2 ):    #периферийный вихрь

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

        sqreps = (1 + eps ** 2)


        rm = ((1 + r2) / 2) + calcul.delta

        rs = np.arange(r2, 1, 0.05)

        r0 = 2 * rm - r2

        for j, theta in enumerate(calcul.theta2):
            m = 1 + ((math.log10(calcul.R1 / calcul.R2)) - (math.log10(theta))) * (1 / (math.log10(r2)))
            calcul.ms.append(m)

        for k, r in enumerate(rs):


            for l, M1 in enumerate(calcul.allM1):
                mz1 = M1 / (sqreps ** 0.5)

                w1 = M1 * calcul.a1

                wz1 = mz1 * calcul.a1

                mf1 = mz1 * calcul.eps1

                wf1 = mf1 * calcul.a1

                a = mz1 / ((1 - r2) * (r2 - 2 * rm + 1))

                b = -2 * a * rm

                c = mz1 - a - b

                n = ((a * (rm - 1)) ** 2) / mz1

                _mzm = 1 + n

                mzm = mz1 * _mzm

                mz = (a * r ** 2) + b * r + c
                mzsPer.append(mz)

                for p, m in enumerate(calcul.ms):

                    A = ((calcul.k1 - 1)/(2 * m)) * ((M1 ** 2)/(1 + sqreps))

                    func1 = (1 - A * ((1 / (r ** (2 * m))) - 1))
                    func2 = np.sqrt(calcul.k1 * calcul.R1 * calcul.T1 * func1)

                    mf = M1 / ((sqreps ** 0.5) * (func1 ** 0.5) * (r ** (2 * m)))
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
    def gamma_calc(r2, m, eps, pi0, fi2, M1):
        sqreps = 1 + eps ** 2
        return (((calcul.k1 * (calcul.k2 - 1)) / (2 * calcul.k2)) * ((calcul.z1 * calcul.R1) / (calcul.z2 * calcul.R2))*
                ((M1 ** 2) / ((r2 ** (2 * m)) * calcul.theta2 * fi2 * sqreps *
                (1 - ((1 / pi0) ** ((calcul.k2 - 1) / calcul.k2)) * (1 / (fi2 ** ((calcul.k1 * (calcul.k2 - 1)) /
                                                                                  (calcul.k2 * (calcul.k1 - 1)))))))))

    #
    # @staticmethod
    # def fi2_calc(r2, A, m):
    #     return 1 - A * ((1 / (r2 ** (2 * m))) - 1)

    # @staticmethod
    # def A_calc(sqreps, M1, k1, ms):
    #
    #     for p, m in enumerate(ms):
    #
    #         A = ((k1 - 1) / (2 * m)) * ((M1 ** 2) / (1 + sqreps))
    #
    #         return A
    #
    # @staticmethod
    # def m_calc(r2, ms):
    #
    #     for j, theta in enumerate(calcul.theta2):
    #         m = 1 + ((math.log10(calcul.R1 / calcul.R2)) - (math.log10(theta))) * (1 / (math.log10(r2)))
    #         ms.append(m)
    #         return ms


    @staticmethod
    def axial( eps, r2):
        #приосевой вихрь

        mzs3zone = []
        mzs4zone = []
        mfsAx = []
        wfsAx = []
        Re = []
        r3zones = []
        r4zones = []
        #
        sqreps = (1 + eps ** 2)

        # for i, r2 in enumerate(r2):

        rs = np.arange(0, r2, 0.05)

        rg = r2

        for o, pi0 in enumerate(calcul.Pi0):

            for j, theta in enumerate(calcul.theta2):

                m = 1 + ((math.log10(calcul.R1/calcul.R2)) - (math.log10(theta))) * (1 / (math.log10(r2)))

                calcul.ms.append(m)

                for l, M1 in enumerate(calcul.allM1):

                    mz1 = M1/(sqreps ** 0.5)

                    w1 = M1 * calcul.a1

                    wz1 = mz1 * calcul.a1

                    mf1 = mz1 * eps

                    Piks = 1/(1 - (((M1 ** 2) * calcul.k1)/2))

                    Pie = Piks

                    for p, m in enumerate(calcul.ms):

                        A = ((calcul.k1 - 1) / (2 * m)) * ((M1 ** 2) / (1 + sqreps))

                        fi2 = 1 - A * ((1 / (r2 ** (2 * m))) - 1)

                        # gamma = ((k1 * (k2 - 1))/(2 * k2)) * ((z1 * R1)/(z2 * R2)) * \
                        #         ((M1 ** 2)/((r2 ** (2 * m)) * theta2 * fi2 * sqreps *
                        #                     (1 - ((1/pi0) ** ((k2 - 1)/k2)) *
                        #                      (1/(fi2 ** ((k1 * (k2 - 1))/(k2 * (k1 - 1))))))))

                        gamma = calcul.gamma_calc( r2, m, eps, pi0, fi2, M1)

                        B = ((calcul.z1 * calcul.R1)/(calcul.z2 * calcul.R2)) * (((calcul.k2 - 1) * calcul.k1 * (M1 ** 2))/
                                                     (2 * gamma * calcul.k2 * sqreps * (r2 ** (2 * m)) * calcul.theta2 * fi2))

                        C = ((calcul.k1 * (calcul.k3 - 1) * (M1 ** 2))/(2 * calcul.k3 * gamma * sqreps * fi2 * (r2 ** (2 * m))))\
                            * ((calcul.z1 * calcul.R1)/(calcul.z3 * calcul.R3))

                        re = rg * ((1 - (1/C) *
                                    (1 - ((1/(Pie * (fi2 ** (calcul.k1/(calcul.k1 - 1))))) ** ((calcul.k3 - 1)/calcul.k3)))) ** 1/(2 * gamma))

                        re = re.tolist()

                        Re.extend(re)


                        for h, re in enumerate(Re):

                            r3zones = np.arange(re, rg, 0.05)

                            # r3zones.append(r3zone)

                            r4zones = np.arange(0, re, 0.05)

                            # r4zones.append(r4zone)

                            for u, r3zone in enumerate(r3zones):

                                ksi3zone = 1 - C * (1 - ((r3zone/rg) ** (2 * gamma)))

                                mz3zone = np.sqrt((2/calcul.k3) * (1 - (1/(Piks * (fi2 ** (calcul.k1/(calcul.k1 - 1))) *
                                                        (calcul.Ksig ** (calcul.k2/(calcul.k2 - 1))) * (ksi3zone ** (calcul.k3/(calcul.k3 - 1)))))))

                                mzs3zone.append(mz3zone)

                            for y, r4zone in enumerate(r4zones):

                                ksi4zone = 1 - C * (1 - ((r4zone/re) ** (2 * gamma))) # Ksi E

                                D = ((calcul.z1 * calcul.R1)/(calcul.ze * re)) * ((re/r2) ** (2 * gamma)) *\
                                                                                    ((calcul.k1 * (calcul.ke - 1) * (M1 ** 2))/
                                     (2 * calcul.ke * gamma * sqreps * fi2 * calcul.Ksig * ksi4zone * (r2 ** (2 * m))))

                                U = 1 - D * (1 - ((r4zone/re) ** (2 * gamma)))

                                mz4zone = np.sqrt((2/calcul.k3) * ((1/(Piks * (fi2 ** (calcul.k1/(calcul.k1 - 1))) * (calcul.Ksig ** (calcul.k2/(calcul.k2 - 1))) * (ksi4zone ** (calcul.k3/(calcul.k3 - 1))) *(U ** (calcul.ke/(calcul.ke - 1))))) - 1))

                                mzs4zone.append(mz4zone)

                            for k, r in enumerate(rs):

                                MfAx = mf1 * (1/(r2 ** m)) * ((r/r2) ** gamma) * \
                                       (((calcul.k1 * calcul.R1)/(calcul.k2 * calcul.R2)) ** 0.5) * \
                                       (1/((fi2 ** 0.5) * ((1 - B * (1 - ((r/r2) ** (2 * gamma)))) ** 0.5)))

                                mfsAx.append(MfAx)

                                WfAx = calcul.a1 * MfAx

                                wfsAx.append(WfAx)
                                print(wfsAx)

                                return wfsAx
    # @staticmethod
    # def run():
    #     calcul.peripheral(calcul.R1, calcul.R2, calcul.delta,
    #                       calcul.eps1, calcul.r2s, calcul.allM1,
    #                       calcul.a1, calcul.theta2, calcul.ms,
    #                       calcul.k1, calcul.T1)


# if __name__ == '__main__':
#
#     # periph = calcul.peripheral(calcul.R1, calcul.R2, calcul.delta,
#     #                       calcul.eps1, calcul.r2s, calcul.allM1,
#     #                       calcul.a1, calcul.theta2, calcul.ms,
#     #                       calcul.k1, calcul.T1)
#
#     axial = calcul.axial(calcul.R1, calcul.R2, calcul.R3, calcul.Re, eps, calcul.r2s,
#                          calcul.allM1, calcul.a1, calcul.theta2, calcul.ms, calcul.k1, calcul.k2,
#                          calcul.k3, calcul.ke, calcul.z1, calcul.z2, calcul.z3, calcul.ze, calcul.Pi0, calcul.Ksig)

    # pool = Pool()
    # pool.map(periph, axial)
    # pool.close()
    # pool.join()


    # calcul.run()
