# import packages
import numpy as np
import scipy.io
import pandas as pd
import pickle
import os
import time
import copy
from pprint import pprint
from tqdm import tqdm
from abc import ABC, abstractmethod

np.set_printoptions(precision=6, suppress=True, formatter={'float': '{: 8.6f}'.format})


# utility function
class CatchTime:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start


class Constant:

    eps = 1e-2
    p_vals = [1.0, 1.5, 2.0]
    path_output = 'output_v1'


# class for generating data as an instance of MESP
class MESProblemInstance:

    def __init__(self, C, s):
        self.d = C.shape[0]
        self.s = s
        self.C = C
        self.varphi = None
        self.str_varphi = ''

    def objective_type(self, str_varphi):
        self.str_varphi = str_varphi
        if str_varphi == 'log':
            self.varphi = lambda x: -np.log(x)
        elif str_varphi == '1/x':
            self.varphi = lambda x: 1./x
        else:
            raise Exception('invalid objective type')

    @classmethod
    def read_data(cls, d, s):
        # read data from file
        C = scipy.io.loadmat(f'data/data{d:d}.mat')['C']
        # C = np.array(C)
        # C = C.reshape(d, d)
        return cls(C, s)

    # @classmethod
    # def read_data_d124(cls, s):
    #     # read data from file
    #     d = 124
    #     C = pd.read_table(os.getcwd() + '/Data124.dms',
    #                       header=None, encoding='utf-8', sep='\s+')
    #     C = np.array(C)
    #     C = C.reshape(d, d)
    #     return cls(C, s)
        
    @classmethod
    def generate_data(cls, d, s, seed):
        # generate random data
        np.random.seed(seed)
        sqrt = np.random.normal(size=(d, d))
        C = sqrt @ sqrt.T
        return cls(C, s)


class PreProcessing:
    def __init__(self, data):
        self.data = data
        try:
            self.V = np.linalg.cholesky(data.C)  # note inconsistency with paper: V is transposed compared to paper
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(data.C)
            # Clamp tiny negative eigenvalues to zero
            eigvals[eigvals < 0] = 0
            # Construct the square-root factor:  A ≈ B @ B.T
            self.V = eigvecs @ np.diag(np.sqrt(eigvals))
        # intermediate quantities used by code by Li and Xie
        self.E = np.eye(data.d, dtype=int)  # identity matrix
        self.V_square = [self.V[i, None].T * self.V[i, None] for i in range(data.d)]  # V[i, None] is row vector
        # used by our complementary problem and/or linx
        self.W = np.linalg.pinv(self.V).T
        self.log_det_C = np.linalg.slogdet(data.C)[1]
        self.g = 1. / np.partition(np.diag(data.C), -data.s)[-data.s]  # taking p = 1 in Anstreicher (2020)
        # objective function
        if data.str_varphi == 'log':
            self.varphi_grad = lambda x: -1. / x
        elif data.str_varphi == '1/x':
            self.varphi_grad = lambda x: -1. / x**2
        else:
            raise Exception('invalid objective type')


class AlgorithmOutput:
    def __init__(self):
        self.primal_val = []
        self.dual_val = []
        self.dual_trajectory = []
        self.trajectory = []
        self.avg_trajectory = []
        self.avg_vals = []
        self.time = []  # 0
        self.num_iter = 0
        self.num_update = 0
        self.sp_gap = []
        self.sp_gap_avg = []
        self.lg_trajectory = []     # trajectory of log of scaling
        self.alpha_trajectory = []  # trajectory of weight coefficients alpha


# class for running an algorithm and save the stats
class MESProblemAlgorithm(ABC):

    def __init__(self, parameters=None):
        self.data = None                                    # problem instance input data
        self.aux = None                                     # auxiliary data calculated from input data
        self.stats = AlgorithmOutput()                      # algorithm output and stats
        self.parameters = parameters \
            if parameters is not None \
            else {}                                         # algorithm parameters
        self.i_iter = -1                                    # current number of iterations
        self.x = None                                       # current iterate
        self.val = None                                     # current objective value
        self.eigen = {}                                     # current eigenvalue decomposition of X(x)
        self.Y = None
        self.gamma = .01
        self.simplex_ent = False
        self.max_iter = 0
        self.G = None

    def preprocess(self, instance):
        self.data = instance
        self.aux = PreProcessing(self.data)
        self.G = np.ones(self.data.d)

    def initialize(self, initial_pt=None):
        if initial_pt is None:
            # self.x = np.concatenate((np.ones(self.data.s), np.zeros(self.data.d - self.data.s)))
            self.x = np.ones(self.data.d) * self.data.s / self.data.d
        else:
            self.x = initial_pt
        self.val = self.obj(self.x)
        self.stats.primal_val.append(self.val)
        self.stats.trajectory.append(copy.deepcopy(self.x))

    def record_stats(self):
        self.stats.primal_val.append(self.val)
        self.stats.trajectory.append(copy.deepcopy(self.x))

    def save_stats(self, foldername, filename_desc='', path='', create_dir=False):
        if len(path) == 0:
            path = Constant.path_output
        directory = os.path.join(path, foldername)
        # if create_dir:
        os.makedirs(directory, exist_ok=True)
        with open(self.get_stats_path(foldername, self.data.d, self.data.s, filename_desc, path), 'wb') as f:
            pickle.dump(self.stats.__dict__, f)

    @classmethod
    def get_stats_path(cls, foldername, d, s, filename_desc='', path=''):
        directory = os.path.join(path, foldername)
        filename = '_'.join([cls.__name__, filename_desc, 'd' + str(d), 's' + str(s)])
        return os.path.join(directory, filename + '.pkl')

    def run(self, instance, initial_pt=None, max_iter=1000, prt=False):

        self.max_iter = max_iter

        if self.aux is None:
            self.preprocess(instance)

        with CatchTime() as timer:

            self.initialize(initial_pt)

            tic = time.perf_counter()

            for self.i_iter in tqdm(range(max_iter), desc=self.__class__.__name__, disable=not prt):
                self.val = self.update()
                self.record_stats()
                if self.stopping_criteria():
                    break

                self.stats.time.append(time.perf_counter() - tic)

        self.stats.num_iter += self.i_iter + 1

    @abstractmethod
    def update(self):
        self.stats.num_update += 1

    def stopping_criteria(self):
        return False
        # if len(self.stats.primal_val) >= 10:
        #     last10 = self.stats.primal_val[-10:]
        #     return np.square(last10 - np.mean(last10)).mean() < Constant.eps
        # else:
        #     return False

    def find_k(self, lmbd, s=None):
        if s is None:
            s = self.data.s
        sum_lmbd = np.sum(lmbd)
        for i in range(s):
            nu = sum_lmbd / (s - i)
            if nu >= lmbd[i] - 1e-10:
                return i, nu
            sum_lmbd -= lmbd[i]

    def x_to_X(self, x):
        # X = self.aux.V.T @ np.diag(x) @ self.aux.V
        X = sum(x[i] * self.aux.V_square[i] for i in range(self.data.d))
        return X

    def obj_matrix(self, X, s, dual_oracle=True):
        lmbd, U = np.linalg.eigh(X)  # eigenvalue decomposition
        lmbd = lmbd[::-1]
        k, nu = self.find_k(lmbd, s)
        # self.eigen['lmbd'] = lmbd
        # self.eigen['k'] = k
        # self.eigen['nu'] = nu
        # self.eigen['U'] = U
        obj_val = 0
        for i in range(k):
            obj_val += self.data.varphi(lmbd[i])
        obj_val += self.data.varphi(nu) * (s - k)
        if dual_oracle:
            y = np.zeros(self.data.d)
            for j in range(k):
                y[-1 - j] = self.aux.varphi_grad(lmbd[j])
            for j in range(k, self.data.d):
                y[-1 - j] = self.aux.varphi_grad(nu)
            self.Y = U @ np.diag(y) @ U.T
        # else:
        #     self.eigen['lmbd'] = lmbd
        #     self.eigen['k'] = k
        #     self.eigen['nu'] = nu
        #     self.eigen['U'] = U
        return obj_val

    def obj(self, x):
        X = self.x_to_X(x)
        return self.obj_matrix(X, self.data.s)

    def proj_x(self, x):
        # make projection onto domain of x
        ext_x = np.concatenate((x, x - 1))
        ind_x = np.argsort(ext_x)
        sorted_x = ext_x[ind_x]
        is_x_minus_1 = np.concatenate((np.zeros(self.data.d), np.ones(self.data.d)))[ind_x]
        sorted_x = np.concatenate((sorted_x, np.array([np.inf])))
        i = 2 * self.data.d - 1
        sum_x = -self.data.s
        denom = 0
        while i >= 0 and not (denom > 0 and sorted_x[i] <= sum_x / denom <= sorted_x[i + 1]):
            if is_x_minus_1[i] == 1:
                sum_x -= sorted_x[i]
                denom -= 1
            else:
                sum_x += sorted_x[i]
                denom += 1
            i -= 1
        nu = sum_x / denom
        res = np.minimum(np.maximum(x - nu, np.zeros(self.data.d)), np.ones(self.data.d))
        return res


if __name__ == '__main__':
    # test = MESProblemInstance.read_data_d124(s=20)
    test = MESProblemInstance.generate_data(d=5, s=3, seed=5)
    test.objective_type('log')
    pprint(vars(test))
    print(test.varphi(np.exp(1.)))
    print()

    prep = PreProcessing(test)
    pprint(vars(prep))
    print(np.linalg.norm(prep.V @ prep.V.T - test.C),  # C = V @ V.T as opposed to the notation in the paper
          len(prep.V_square), prep.V_square[0].shape,
          np.linalg.norm(prep.W.T @ prep.V - prep.E),
          prep.log_det_C - np.log(np.linalg.det(test.C)),
          prep.varphi_grad(1),
          sep='\n')
    print()

    # algo = MESProblemAlgorithm()
    # algo.preprocess(instance=test)
    # # algo.run(test, max_iter=11)
    # pprint(algo.stats.__dict__)
    # pprint(algo.__dict__)

    # # testing find_k
    # x = np.array([5, 4, 3, 2, 1])
    # print(algo.find_k(x))
    # x = np.array([5, 4, 3, 0, 0])
    # print(algo.find_k(x))
    # # testing obj
    # x = np.array([1, 1, 1, 0, 0])
    # print(algo.obj(x) + np.sum(np.log(np.linalg.eigh(test.C[:3, :3])[0][-3:])))
    #
    # # # testing save_stats
    # # algo.save_stats(directory='test', create_dir=True)
    # # with open(os.path.join(Constant.path_output, 'test', 'MESProblemAlgorithm_d5_s3.pkl'), 'rb') as f:
    # #     loaded_stats = pickle.load(f)
    # # pprint(loaded_stats)
