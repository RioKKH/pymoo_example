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
from pymoo.util.misc import stack
from pymoo.indicators.hv import Hypervolume


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


class HisTestProblem(HisProblem):

    def _calc_pareto_front(self, flatten=True, *args, **kwargs):
        f2 = lambda f1: ((f1/100) ** 0.5 - 1)**2
        F1_a, F1_b = np.linspace(1, 16, 300), np.linspace(36, 81, 300)
        F2_a, F2_b = f2(F1_a), f2(F1_b)

        pf_a = np.column_stack([F1_a, F2_a])
        pf_b = np.column_stack([F1_b, F2_b])

        return stack(pf_a, pf_b, flatten=flatten)

    def _calc_pareto_set(self, flatten=True, *args, **kwargs):
        x1_a = np.linspace(0.1, 0.4, 50)
        x1_b = np.linspace(0.6, 0.9, 50)
        x2 = np.zeros(50)

        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
        return stack(a,b, flatten=flatten)


def main():
    # problem = HisProblem()
    problem = HisTestProblem()

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

    # X = res.X
    # F = res.F
    X, F = res.opt.get("X", "F")
    hist = res.history
    print(len(hist))

    n_evals = []         # corresponding number of function evaluations
    hist_F = []          # the objective space values in each generation
    hist_cv = []         # constraint violation in each generation
    hist_cv_avg = []     # average constraint violation in the whole population

    for algo in hist:

        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least constraint violation and the averate in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    k = np.where(np.array(hist_cv) <= 0.0)[0].min()
    print(f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations.")
    # replace this line by "hist_cv" if you like to analyze the least 
    # feasible optimal solution and not the population
    vals = hist_cv_avg

    k = np.where(np.array(vals) <= 0.0)[0].min()
    print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, vals, color="black", lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, vals, facecolor="none", edgecolor="black", marker="p")
    plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.show()

    #fl = F.min(axis=0)
    #fu = F.max(axis=0)
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    metric = Hypervolume(ref_point = np.array([1.1, 1.1]),
                         norm_ref_point = False,
                         zero_to_one = True,
                         ideal = approx_ideal,
                         nadir = approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color="black", lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv, facecolor="none", edgecolor="black", marker="p")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()
    #pf_a, pf_b = problem.pareto_front(use_cache=False, flatten=False) 
    #pf = problem.pareto_front(use_cache=False, flatten=True)

    #xl, xu = problem.bounds()
    #plt.figure(figsize=(7, 5))
    #plt.scatter(X[:, 0], X[:, 1], s=30, facecolors="none", edgecolors="r")
    #plt.xlim(xl[0], xu[0])
    #plt.ylim(xl[1], xu[1])
    #plt.title("Design Space")
    #plt.show()

    #print(f"Scale f1: [{fl[0]}, {fu[0]}]")
    #print(f"Scale f2: [{fl[1]}, {fu[1]}]")
    #plt.figure(figsize=(7, 5))
    #plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue",
    #            label="Solutions")
    #plt.plot(pf_a[:, 0], pf_a[:, 1], alpha=0.5, linewidth=2.0, color="red",
    #         label="Pareto-front")
    #plt.plot(pf_b[:, 0], pf_b[:, 1], alpha=0.5, linewidth=2.0, color="red")
    #plt.scatter(approx_ideal[0], approx_ideal[1], facecolors="none",
    #            edgecolors="red", marker="*", s = 100,
    #            label="Ideal Point (Approx)")
    #plt.scatter(approx_nadir[0], approx_nadir[1], facecolors="none",
    #            edgecolors="black", marker="p", s=100,
    #            label="Nadir Point (Approx)")
    #plt.title("Objective Space")
    #plt.legend()
    #plt.show()



