#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pymoo.problems import get_problem
from pymoo.util.plotting import plot_problem_surface, plot
from pymoo.visualization.scatter import Scatter

def single():
    problem = get_problem("ackley", n_var=2, a=20, b=1/5, c=2*np.pi)
    plot_problem_surface(problem, 100, plot_type="wireframe+contour")

    problem = get_problem("rastrigin", n_var=2)
    plot_problem_surface(problem, 100, plot_type="wireframe+contour")

    problem = get_problem("sphere", n_var=2)
    plot_problem_surface(problem, 100, plot_type="wireframe+contour")

    problem = get_problem("zakharov", n_var=2)
    plot_problem_surface(problem, 100, plot_type="wireframe+contour")


def multi():
    problem = get_problem("zdt1")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("zdt2")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("zdt3")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("zdt4")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("zdt5")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("zdt6")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("bnh")
    plot(problem.pareto_front(), no_fill=True)

    problem = get_problem("rosenbrock", n_var=2)
    plot_problem_surface(problem, 100, plot_type="wireframe+contour")

    problem = get_problem("griewank", n_var=2)
    plot_problem_surface(problem, 100, plot_type="wireframe+contour")

def multi2():
    pf = get_problem("truss2d").pareto_front()

    plot = Scatter(title="Pareto-front")
    plot.add(pf, s=80, facecolors='none', edgecolors='r')
    plot.add(pf, plot_type="line", color="black", linewidth=2)
    plot.show()

    plot.reset()
    plot.do()
    plot.apply(lambda ax: ax.set_yscale("log"))
    plot.apply(lambda ax: ax.set_xscale("log"))
    plot.show()


if __name__ == '__main__':
    # multi()
    multi2()

