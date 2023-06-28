#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.factory import get_sampling
from pymoo.factory import get_crossover
from pymoo.factory import get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


# 問題の定義
problem = get_problem("zdt1")

# アルゴリズムの定義
algorithms = NSGA2(
    pop_size = 100, # 集団のサイズ
    n_offsprings = 100, # 1世代あたりの子供の数
    sampling = get_sampling("real_random"), # 初期サンプリング方法
    crossover=get_crossover("real_sbx", prob=0.9, eta=15), # 交叉方法
    mutation = get_mutation("real_pm", eta=20), # 突然変異法
    eliminate_duplicates=True #重複個体の削除を行うかどうか
)

# 最適化の実行
res = minimize(
    problem,
    algorithms,
    ("n_gen", 200),
    verbose=True,
)

# 結果を可視化する
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
