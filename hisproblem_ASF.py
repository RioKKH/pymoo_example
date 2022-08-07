#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights


class HisProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2, # number of constraints of inequality
                         xl=np.array([-2, -2]),
                         xu=np.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

def main():
    problem = HisProblem()
    algolithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 40)
    res = minimize(problem,
                   algolithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    fl = nF.min(axis=0)
    fu = nF.max(axis=0)
    weights = np.array([0.2, 0.8])

    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors="none", edgecolors="r")
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.scatter(approx_ideal[0], approx_ideal[1], facecolors="none",
                edgecolors="red", marker="*", s = 100,
                label="Ideal Point (Approx)")
    plt.scatter(approx_nadir[0], approx_nadir[1], facecolors="none",
                edgecolors="black", marker="p", s=100,
                label="Nadir Point (Approx)")
    plt.title("Objective Space")
    plt.legend()
    plt.show()

    print(f"Scale f1: [{fl[0]}, {fu[0]}]")
    print(f"Scale f2: [{fl[1]}, {fu[1]}]")
    plt.figure(figsize=(7, 5))
    plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.title("Objective Space")
    plt.show()

    decomp = ASF()
    i = decomp.do(nF, 1/weights).argmin()
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
    plt.title("Objective Space")
    plt.show()

    i = PseudoWeights(weights).do(nF)
    print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
    plt.title("Objective Space")
    plt.show()




