"""
Run all option-implied moment estimation methods.
"""

import plot_comparison
from method_direct_moments_bkm import run_method as run_direct_bkm

#
from method_main import run_method as run_nonparametric
from method_parametric_lognormal import run_method as run_parametric_lognormal
from method_parametric_normal import run_method as run_parametric_normal

if __name__ == "__main__":
    run_nonparametric()
    run_parametric_normal()
    run_parametric_lognormal()
    run_direct_bkm()
    plot_comparison.run_plot()
    import diagnose_divergences
