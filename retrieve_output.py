import matplotlib.pyplot as plt

from gamma import *
from linx import *
from base import Constant
from constants import get_algo_names, get_s_range, get_opt


# create tables for retrieving stats for varying s and algorithms
stats = {}

# specify filename of saved output
Constant.path_output = 'output_SC'
# foldername = '20251026-113944-d90'
foldername = '20251026-113339-d124'

# retrieve config of saved run
d = int(foldername.split('-d')[-1])
s_range = get_s_range(d)
opt = get_opt(d)

# algorithms tested
linx_algos = [OScaledLinxExtragradient, GScaledLinxExtragradient, GGScaledLinxExtragradient]
gamma_algos = [GammaMirrorDescentAlgorithm, GammaComplMirrorDescentAlgorithm, GammaStarExtragradientAlgorithm]
algos = linx_algos + gamma_algos
algo_names = get_algo_names(algos)

# retrieve data from saved table
for key in ['time', 'gap', 'lb']:
    stats[key] = np.genfromtxt(os.path.join(Constant.path_output, foldername, key + '.csv'), delimiter=',')[:, 1:]

# display the results as a formatted table
# header of the table
print('\\toprule')
print(' & '.join([''] + [f'\\multicolumn{{4}}{{c|}}{{{st}}}' for st in algo_names])
      + ' \\\\')
print(' & '.join(['s'] + ['LB', 'gap', 'conv.', 'time'] * len(algos)) + ' \\\\')
print('\\midrule')
# data of the table
for j, s in enumerate(s_range):
    print(f'{s:4d} & ', end='')
    line = []
    for i in range(len(algos)):
        line += [f'{stats["lb"][j][i]:8.3f}',
                 f'{-opt[s] - stats["lb"][j][i] :.3f}',
                 f'{stats["gap"][j][i]:.4f}',
                 f'{stats["time"][j][i]:5.2f}']
    print(' & '.join(line) + ' \\\\')
print('\\bottomrule')

# convergence plot
plt.style.use({'font.family': 'Arial', 'font.size': 20})
plt.figure(figsize=(11, 6))
# plot the figure for optimality gap
for i in range(len(algos)):
    # gap_list = [- opt[s] - stats["lb"][j][i] for j, s in enumerate(s_range)]
    gap_list = [stats["time"][j][i] for j, s in enumerate(s_range)]
    plt.plot(s_range, gap_list, marker='o')
plt.ylim(bottom=0)
plt.legend(algo_names, loc='center left', bbox_to_anchor=(1, 0.5))
# plt.title(f'd = {d}')  # f'gap, d = {d}')
# plt.ylabel(f'integrality gap', fontsize=25)
plt.ylabel(f'time', fontsize=25)
plt.xlabel(f'subset size', fontsize=25)
plt.tight_layout()  # Prevent label cutoff

# save figure to file
plt.savefig('output/time.pdf', bbox_inches="tight")
plt.show()