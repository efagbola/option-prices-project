"""
Run all option-implied moment estimation methods.
"""

from method_main import run_method as run_nonparametric
from method_parametric_normal import run_method as run_parametric_normal
from method_parametric_lognormal import run_method as run_parametric_lognormal
from method_direct_moments_bkm import run_method as run_direct_bkm


if __name__ == "__main__":
    run_nonparametric()
    run_parametric_normal()
    run_parametric_lognormal()
    run_direct_bkm()
