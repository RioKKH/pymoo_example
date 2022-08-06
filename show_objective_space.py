#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(7, 5))

    f2 = lambda f1: - ((f1/100)**0.5 - 1)**2
    F1_a, F1_b = np.linspace(1, 16, 300), np.linspace(36, 81, 300)
    F2_a, F2_b = f2(F1_a), f2(F1_b)

    plt.rc("font", family="serif")
    plt.plot(F1_a, F2_a, linewidth=2.0, color="green", label="Pareto-front")
    plt.plot(F1_b, F2_b, linewidth=2.0, color="green")

    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")

    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.10),
               ncol=4,
               fancybox=True,
               shadow=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
