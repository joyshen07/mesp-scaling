from scaling_linx_bt import *
from scaling_gamma import *

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')


# experiment setup
# d = 124
# # s = 50
# s_range = [s for s in range(20, 110, 10)]
d = 90
s_range = [s for s in range(20, 90, 10)]
d = 2000
s_range = [100, 200, 300, 500, 700, 900, 1000]

# algorithms tested
linx_algos = [OScaledLinxExtragradient, GScaledLinxExtragradient, GGScaledLinxExtragradient]
gamma_algos = [GammaMirrorDescentAlgorithm, GammaComplMirrorDescentAlgorithm]
algos = linx_algos  # + gamma_algos + [GammaStarExtragradientAlgorithm]
# algos = [GammaMirrorDescentAlgorithm, ScaledGammaExtragradientAlgorithm]
# algos = [GammaComplMirrorDescentAlgorithm, ScaledGammaComplExtragradientAlgorithm]
# algorithm names
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
algo_names = [algo_name_mapping[algo] for algo in algos]

# create tables for saving stats for varying s and algorithms
# lower bounds, convergence measurement (gradient-related), running time
stats = {}
for key in ['lb', 'gap', 'time', 'iter']:
    stats[key] = [[0] * len(algos) for _ in s_range]

# varying s
for j, s in enumerate(s_range):

    # build the data matrix for testing
    test = MESProblemInstance.read_data(d=d, s=s)
    test.objective_type('log')
    print('\n' + '-' * 20)
    print(f'd = {d}, s = {s}')

    # run each algorithm
    for i, Alg in enumerate(algos):
        alg = Alg()
        alg.run(instance=test, max_iter=1000, prt=True)     # number of iterations fixed to 1000

        # save running time
        stats['time'][j][i] = alg.stats.time[-1]

        # save number of iterations
        stats['iter'][j][i] = alg.i_iter + 1

        # compute a valid lower bound
        lb = 0
        if Alg in linx_algos:
            # retrieve the solution output x, lg
            x = alg.stats.trajectory[-1]
            lg = alg.stats.lg_trajectory[-1]
            # compute by minimizing the first order approximation over x domain
            grad_x, _ = alg.grad_z(x, lg)
            obj_val = alg.obj_slinx(x, lg)
            x_tmp = np.zeros(alg.data.d)
            x_tmp[np.argpartition(grad_x, alg.data.s)[:alg.data.s]] = 1  # smallest s elements
            lb = obj_val + grad_x @ (x_tmp - x)
        elif Alg in [ScaledGammaExtragradientAlgorithm, ScaledGammaComplExtragradientAlgorithm]:
            # retrieve the solution output x, lg
            x = alg.stats.trajectory[-1]
            lg = alg.stats.lg_trajectory[-1]
            # compute by minimizing the first order approximation over x domain
            grad_x, _, obj_val = alg.grad_z(x, lg)
            x_tmp = np.zeros(alg.data.d)
            x_tmp[np.argpartition(grad_x, alg.data.s)[:alg.data.s]] = 1  # smallest s elements
            lb = obj_val + grad_x @ (x_tmp - x)
        elif Alg in gamma_algos:
            x = alg.stats.trajectory[-1]
            # compute by minimizing the first order approximation over x domain
            obj, grad = alg.obj_n_grad(x) if Alg == GammaMirrorDescentAlgorithm else alg.obj_n_grad_compl(x)
            x_tmp = np.zeros(alg.data.d)
            x_tmp[np.argpartition(grad, alg.data.s)[:alg.data.s]] = 1  # smallest s elements
            lb = obj + grad @ (x_tmp - x)
        elif Alg == GammaStarExtragradientAlgorithm:
            # retrieve the solution output x, alpha
            x = alg.stats.trajectory[-1]
            alpha = alg.stats.alpha_trajectory[-1]
            # compute by minimizing the first order approximation over x domain
            grad_x, objs = alg.grad_z(x, alpha)
            x_tmp = np.zeros(alg.data.d)
            x_tmp[np.argpartition(grad_x, alg.data.s)[:alg.data.s]] = 1  # smallest s elements
            lb = alpha @ objs + grad_x @ (x_tmp - x)
            # compute upper bound so that saddle point gap = ub - lb in the next step
            obj = max(objs)
        else:
            raise Exception('unknown algorithm: cannot compute a lower bound')
        print(f'lower bound = {lb}')
        # save lower bound
        stats['lb'][j][i] = lb

        # compute indicator of convergence
        if Alg in linx_algos or Alg in [ScaledGammaExtragradientAlgorithm, ScaledGammaComplExtragradientAlgorithm]:
            # display the norm of grad + xi which indicates quality of convergence
            print(f'||grad + xi|| = {np.linalg.norm(alg.grad_xi)}')
            # save the convergence quality measurement
            stats['gap'][j][i] = np.linalg.norm(alg.grad_xi)
        elif Alg in gamma_algos or Alg == GammaStarExtragradientAlgorithm:
            # display obj - lb indicates quality of convergence
            print(f'obj - lb = {obj - lb}')
            # save the convergence quality measurement
            stats['gap'][j][i] = obj - lb
        else:
            raise Exception('unknown algorithm: cannot compute an indicator of convergence')

# reference: optimal value for d=124 instances from [Anstreicher, 2020]
opt = {
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

# display the results as a formatted table
# header of the table
print('\\toprule')
print(' & '.join([''] + [f'\\multicolumn{{5}}{{c|}}{{{st}}}' for st in algo_names])
      + ' \\\\')
print(' & '.join(['s'] + ['LB', 'gap', 'conv.', 'time', '\\# iter.'] * len(algos)) + ' \\\\')
print('\\midrule')
# data of the table
for j, s in enumerate(s_range):
    print(f'{s:4d} & ', end='')
    line = []
    for i in range(len(algos)):
        line += [f'{stats["lb"][j][i]:8.3f}',
                 f'{-opt[s] - stats["lb"][j][i] :.3f}',
                 f'{stats["gap"][j][i]:.4f}',
                 f'{stats["time"][j][i]:5.2f}',
                 f'{stats["iter"][j][i]:4d}']
    print(' & '.join(line) + ' \\\\')
print('\\bottomrule')

# # plot the figure for optimality gap
# for i in range(len(algos)):
#     gap_list = [- opt[s] - stats["lb"][j][i] for j, s in enumerate(s_range)]
#     plt.plot(s_range, gap_list, marker='o')
# plt.ylim(bottom=0)
# plt.legend(algo_names, loc='upper right')
# plt.title(f'gap, d = {d}')
# plt.show()

# save the tables & figure
Constant.path_output = 'output_SC'
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
foldername = timestamp + f'-d{d}'
os.mkdir(os.path.join(Constant.path_output, foldername))
for key in ['lb', 'gap', 'time', 'iter']:
    table = np.hstack((np.array([s_range]).T, stats[key]))
    np.savetxt(os.path.join(Constant.path_output, foldername, key + '.csv'), table, delimiter=',',
               header=','.join(['s'] + algo_names))
# plt.savefig(os.path.join(Constant.path_output, foldername, 'gap.pdf'), bbox_inches="tight")

# retrieve saved table
Constant.path_output = 'output_SC'
# foldername = '20250129-120658-d90'
foldername = '20250124-000806-d124'
foldername = '20251026-113339-d124'
foldername = '20251026-113944-d90'
foldername = '20251026-164929-d2000'
for key in ['time', 'gap', 'lb', 'iter']:
    stats[key] = np.genfromtxt(os.path.join(Constant.path_output, foldername, key + '.csv'), delimiter=',')[:, 1:]
# display the results as a formatted table
# header of the table
print('\\toprule')
print(' & '.join([''] + [f'\\multicolumn{{4}}{{c|}}{{{st}}}' for st in algo_names])
      + ' \\\\')
print(' & '.join(['s'] + ['LB', 'gap', 'conv.', 'time', '\\# iter.'] * len(algos)) + ' \\\\')
print('\\midrule')
# data of the table
for j, s in enumerate(s_range):
    print(f'{s:4d} & ', end='')
    line = []
    for i in range(3):  # len(algos)):
        line += [f'{stats["lb"][j][i]:8.3f}',
                 # f'{-opt[s] - stats["lb"][j][i] :.3f}',
                 f'{stats["gap"][j][i]:.4f}',
                 f'{stats["time"][j][i]:5.2f}',
                 f'{int(stats["iter"][j][i]):4d}']
    print(' & '.join(line) + ' \\\\')
print('\\bottomrule')

plt.style.use({'font.family': 'Arial', 'font.size': 20})
plt.figure(figsize=(8, 6))
# plot the figure for optimality gap
for i in range(len(algos)):
    # gap_list = [- opt[s] - stats["lb"][j][i] for j, s in enumerate(s_range)]
    # gap_list = [stats["lb"][j][i] for j, s in enumerate(s_range)]
    gap_list = [stats["time"][j][i] for j, s in enumerate(s_range)]
    plt.plot(s_range, gap_list, marker='o')
plt.ylim(bottom=0)
# plt.legend(algo_names)  # , loc='center left', bbox_to_anchor=(1, 0.5))
# plt.title(f'd = {d}')  # f'gap, d = {d}')
# plt.ylabel(f'integrality gap', fontsize=25)
# plt.ylabel(f'relaxation bound', fontsize=25)
plt.ylabel(f'time', fontsize=25)
plt.xlabel(f'subset size', fontsize=25)
plt.tight_layout()  # Prevent label cutoff
plt.show()
plt.savefig('gap.pdf', bbox_inches="tight")
