from gamma import *
from linx import *


algo_name_mapping = {
    OScaledLinxExtragradient: 'linx o-scaling',
    GScaledLinxExtragradient: 'linx g-scaling',
    GGScaledLinxExtragradient: 'linx double-scaling',
    GammaMirrorDescentAlgorithm: r'$\Gamma$',
    GammaComplMirrorDescentAlgorithm: r'$\Gamma^c$',
    GammaStarExtragradientAlgorithm: r'$\Gamma^*$',
    ScaledGammaExtragradientAlgorithm: r'g-$\Gamma$',
    ScaledGammaComplExtragradientAlgorithm: r'g-$\Gamma^c$'
}


def get_algo_names(algos):
    return [algo_name_mapping[algo] for algo in algos]


def get_s_range(d):
    return [s for s in range(20, 110, 10)] if d == 124 else \
        [s for s in range(20, 90, 10)]


# reference: optimal value for d=124 instances from [Anstreicher, 2020]
def get_opt(d):
    return {
        20: 77.827,
        30: 106.700,
        40: 131.055,
        50: 149.498,
        60: 164.012,
        70: 172.528,
        80: 175.091,
        90: 171.262,
        100: 162.865,
        110: 147.933
    } if d == 124 else {
        10: 58.32,
        20: 111.482,
        30: 161.539,
        40: 209.969,
        50: 257.160,
        60: 303.019,
        70: 347.471,
        80: 389.997
    }
