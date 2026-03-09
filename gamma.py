import numpy as np

from base import *


class GammaMirrorDescentAlgorithm(MESProblemAlgorithm):

    def __init__(self):
        super().__init__()
        self.stepsize0 = None       # stepsize parameter
        # self.dual = None

    def initialize(self, initial_pt=None):
        super().initialize(initial_pt)
        # compute stepsize parameter for dynamic scheme to optimize convergence gap
        # see [Juditsky and Nemirovski, 2011] Proposition 1.1(i) eq.(1.9)
        self.stepsize0 = np.sqrt(min(self.data.s * 2, self.data.d) / np.log(self.max_iter + 2))

    # def stopping_criteria(self):
    #     return self.val - self.dual < 1e-3

    def obj_n_grad_core(self, X, s):
        # compute gradient of Gamma function wrt matrix X
        # compute objective at the same time to avoid repeated computation
        lmbd, U = np.linalg.eigh(X)  # eigenvalue decomposition
        lmbd = lmbd[::-1]
        k, nu = self.find_k(lmbd, s)
        obj_val = 0
        for i in range(k):
            obj_val += self.data.varphi(lmbd[i])
        obj_val += self.data.varphi(nu) * (s - k)
        y = np.zeros(self.data.d)
        for j in range(k):
            y[-1 - j] = self.aux.varphi_grad(lmbd[j])
        for j in range(k, self.data.d):
            y[-1 - j] = self.aux.varphi_grad(nu)
        Y = U @ np.diag(y) @ U.T
        # self.dual = s + np.sum(np.log(-y[-s:]))
        return obj_val, Y

    def obj_n_grad(self, x):
        # compute gradient of Gamma function wrt decision variable x
        # compute objective at the same time to avoid repeated computation
        obj_val, Y = self.obj_n_grad_core(self.x_to_X(x), self.data.s)
        grad = np.diag(self.aux.V @ Y @ self.aux.V.T)
        return obj_val, grad

    def update(self):

        # gradient computation
        # compute objective at the same time to avoid repeated computation
        obj_val, grad = self.obj_n_grad(self.x)

        # stepsize computation: dynamic scheme
        # see [Juditsky and Nemirovski, 2011] Proposition 1.1(i)
        stepsize = self.stepsize0 / np.sqrt(self.i_iter + 1) / np.linalg.norm(grad)

        # gradient update
        self.x = self.proj_x(self.x - stepsize * grad)

        # self.dual += np.sum(np.partition(grad, self.data.s)[:self.data.s])
        # print(self.i_iter, obj_val - self.dual)

        return obj_val  # one iteration behind to avoid repeated calculation


class GammaComplMirrorDescentAlgorithm(GammaMirrorDescentAlgorithm):

    def x_to_X_compl(self, x):
        # compute matrix X for the complementary problem
        X = self.aux.W.T @ np.diag(1 - x) @ self.aux.W
        return X

    def obj_n_grad_compl(self, x):
        # compute gradient of Gamma function wrt decision variable x in the complementary problem
        # compute objective at the same time to avoid repeated computation
        obj_val, Y = self.obj_n_grad_core(self.x_to_X_compl(x), self.data.d - self.data.s)
        grad = -np.diag(self.aux.W @ Y @ self.aux.W.T)
        return obj_val - self.aux.log_det_C, grad

    def update(self):

        # gradient computation
        # compute objective at the same time to avoid repeated computation
        obj_val, grad = self.obj_n_grad_compl(self.x)

        # stepsize computation: dynamic scheme
        # see [Juditsky and Nemirovski, 2011] Proposition 1.1(i)
        stepsize = self.stepsize0 / np.sqrt(self.i_iter + 1) / np.linalg.norm(grad)

        # gradient update
        self.x = self.proj_x(self.x - stepsize * grad)

        # self.dual += np.sum(np.partition(-grad, self.data.d - self.data.s)[:self.data.d - self.data.s])
        # self.dual -= self.aux.log_det_C

        return obj_val  # one iteration behind to avoid repeated calculation


class GammaStarExtragradientAlgorithm(GammaComplMirrorDescentAlgorithm):

    def __init__(self):
        super().__init__()
        self.alpha = np.array([.5, .5])
        # self.dual = None
        self.stepsize = .1          # initial value of stepsize
        self.grad = None            # [grad_x, grad_alpha]
        self.L = 1.                 # initial value of local Lipschitz constant
        self.eps = 1e-10            # tolerance to prevent division by 0
        self.stop = False           # whether stopping criterion is satisfied

    def initialize(self, initial_pt=None):
        super().initialize(initial_pt)
        self.stats.alpha_trajectory.append(self.alpha)      # record alpha in addition to x
        self.grad = self.grad_z(self.x, self.alpha)         # compute initial gradient

    def proj_alpha(self, alp, g):
        alpha = alp + g
        # make projection onto simplex
        sorted_alpha = np.sort(alpha)[::-1]
        m = alpha.size
        i = m - 1
        sum_alpha = np.sum(sorted_alpha) - 1
        denom = m
        while i >= 0 and sum_alpha / denom >= sorted_alpha[i]:
            sum_alpha -= sorted_alpha[i]
            denom -= 1
            i -= 1
        nu = sum_alpha / denom
        res = np.maximum(alpha - nu, np.zeros(m))
        return res

    def grad_z(self, x, alpha):
        # compute the gradient w.r.t. x and alpha
        # grad_x = alpha @ grads
        # grad_alpha = objs
        obj_n_grads = [func(x) for func in [self.obj_n_grad, self.obj_n_grad_compl]]
        objs = np.array([tpl[0] for tpl in obj_n_grads])
        grads = np.array([tpl[1] for tpl in obj_n_grads])
        return alpha @ grads, objs

    def stepsize_update(self):
        # inherited by other extragradient algorithms
        theta = .5
        if self.i_iter >= 100:
            self.stepsize = min(self.stepsize, theta / self.L)
        elif self.i_iter >= 0:
            # initial phase: not enforcing monotonicity
            self.stepsize = theta / self.L

    def update(self):

        # update stepsize
        self.stepsize_update()

        # 1st gradient descent in x
        x_tmp = self.proj_x(self.x - self.stepsize * self.grad[0])
        # 1st gradient ascent in alpha
        alpha_tmp = self.proj_alpha(self.alpha, self.stepsize * self.grad[1])

        # 2nd gradient descent in x & ascent in alpha
        grad_x_tmp, grad_alpha_tmp = self.grad_z(x_tmp, alpha_tmp)
        x_new = self.proj_x(self.x - self.stepsize * grad_x_tmp)
        alpha_new = self.proj_alpha(self.alpha, self.stepsize * grad_alpha_tmp)

        # update local Lipschitz constant
        # stack the primal and dual variables into one vector
        z = np.hstack([self.x, self.alpha])
        z_tmp = np.hstack([x_tmp, alpha_tmp])
        z_new = np.hstack([x_new, alpha_new])
        # stop if the iterates are too close to compute the local Lipschitz constants
        if np.linalg.norm(z_tmp - z) < self.eps or np.linalg.norm(z_tmp - z_new) < self.eps:
            self.stop = True
            return self.stats.primal_val[-1]
        # stack the primal and dual gradients into one vector
        grad_z = np.hstack([self.grad[0], self.grad[1]])
        grad_z_tmp = np.hstack([grad_x_tmp, grad_alpha_tmp])
        grad_x_new, grad_alpha_new = self.grad_z(x_new, alpha_new)
        grad_z_new = np.hstack([grad_x_new, grad_alpha_new])
        # update local Lipschitz constant
        self.L = max(np.linalg.norm(grad_z_tmp - grad_z) / np.linalg.norm(z_tmp - z),
                     np.linalg.norm(grad_z_tmp - grad_z_new) / np.linalg.norm(z_tmp - z_new))

        # update decision variables
        self.x = x_new
        self.alpha = alpha_new
        self.grad = [grad_x_new, grad_alpha_new]

        # if self.i_iter % 50 == 0:
        #     self.record_sp_gap()

        # update objective value
        # grad_alpha = objs
        obj_new = self.alpha @ self.grad[1]
        return obj_new

    def stopping_criteria(self):
        # inherited by other scaling relaxations with same stopping criterion
        return self.stop

    def record_stats(self):
        # inherited by other scaling relaxations using alpha
        super().record_stats()
        self.stats.alpha_trajectory.append(copy.deepcopy(self.alpha))

    def record_sp_gap(self):
        _, objs = self.grad_z(self.x, self.alpha)
        self.stats.sp_gap.append(max(objs) - self.dual_prob(self.alpha, self.x))

    def dual_prob(self, alpha, x0):
        grad_x, objs = self.grad_z(x0, alpha)
        x = np.zeros(self.data.d)
        x[np.argpartition(grad_x, self.data.s)[:self.data.s]] = 1
        return objs @ alpha + (x - x0) @ grad_x

    # def stopping_criteria(self):
    #     return self.stats.sp_gap[-1] < 1e-3


class ScaledGammaExtragradientAlgorithm(MESProblemAlgorithm):

    def __init__(self):
        super().__init__()
        self.lg = 0             # log of scaling factor
        self.L = 1.             # initial value of local Lipschitz constant
        self.stepsize = .1      # initial value of stepsize
        self.grad = None        # [grad_x, grad_lg]
        self.eps = 1e-10        # tolerance to prevent division by 0
        self.grad_xi = None     # a measurement of convergence / saddle point gap
        self.stop = False       # whether stopping criterion is satisfied

    def initialize(self, initial_pt=None):
        super().initialize(initial_pt)
        self.lg = np.zeros(self.data.d)                 # initialize log of scaling vector
        self.stats.lg_trajectory = [self.lg]            # record lg
        grad_x, grad_lg, obj_val = self.grad_z(self.x, self.lg)
        self.grad = [grad_x, grad_lg]                   # compute initial gradient
        self.val = obj_val                              # redefine initial objective
        self.stats.primal_val = [self.val]              # re-record objective

    def obj_n_grad_core(self, X, s):
        # compute gradient of Gamma function wrt matrix X
        # compute objective at the same time to avoid repeated computation
        lmbd, U = np.linalg.eigh(X)  # eigenvalue decomposition
        lmbd = lmbd[::-1]
        k, nu = self.find_k(lmbd, s)
        obj_val = 0
        for i in range(k):
            obj_val += self.data.varphi(lmbd[i])
        obj_val += self.data.varphi(nu) * (s - k)
        y = np.zeros(self.data.d)
        for j in range(k):
            y[-1 - j] = self.aux.varphi_grad(lmbd[j])
        for j in range(k, self.data.d):
            y[-1 - j] = self.aux.varphi_grad(nu)
        Y = U @ np.diag(y) @ U.T
        # self.dual = s + np.sum(np.log(-y[-s:]))
        return obj_val, Y

    def x_to_X_2(self, x, lg):
        xg = x * np.exp(lg)
        return sum(xg[i] * self.aux.V_square[i] for i in range(self.data.d))

    def grad_z(self, x, lg):
        # gradient wrt (x, lg)
        obj_val, Y = self.obj_n_grad_core(self.x_to_X_2(x, lg), self.data.s)
        grad_xg = np.diag(self.aux.V @ Y @ self.aux.V.T)
        g = np.exp(lg)
        grad_x = grad_xg * g + lg  # Y already takes the minus sign into account
        grad_lg = grad_xg * x * g + x
        obj_val += x @ lg
        return grad_x, grad_lg, obj_val

    def stepsize_update(self):
        # inherited by other scaling relaxations
        theta = .5
        if self.i_iter >= 100:
            self.stepsize = min(self.stepsize, theta / self.L)
        elif self.i_iter >= 0:
            # initial phase: not enforcing monotonicity
            self.stepsize = theta / self.L

    def update(self):

        # update stepsize
        self.stepsize_update()

        # 1st gradient update
        x_tmp = self.proj_x(self.x - self.stepsize * self.grad[0])
        lg_tmp = self.lg + self.stepsize * self.grad[1]

        # 2nd gradient update
        grad_x_tmp, grad_lg_tmp, _ = self.grad_z(x_tmp, lg_tmp)
        x_new = self.proj_x(self.x - self.stepsize * grad_x_tmp)
        lg_new = self.lg + self.stepsize * grad_lg_tmp

        # update local Lipschitz constant
        # stack the primal and dual variables into one vector
        z = np.hstack([self.x, self.lg])
        z_tmp = np.hstack([x_tmp, lg_tmp])
        z_new = np.hstack([x_new, lg_new])
        # stop if the iterates are too close to compute the local Lipschitz constants
        if np.linalg.norm(z_tmp - z) < self.eps or np.linalg.norm(z_tmp - z_new) < self.eps:
            self.stop = True
            return self.stats.primal_val[-1]
        # stack the primal and dual gradients into one vector
        grad_z = np.hstack([self.grad[0], self.grad[1]])
        grad_z_tmp = np.hstack([grad_x_tmp, grad_lg_tmp])
        grad_x_new, grad_lg_new, obj_val = self.grad_z(x_new, lg_new)
        grad_z_new = np.hstack([grad_x_new, grad_lg_new])
        # update local Lipschitz constant
        self.L = max(np.linalg.norm(grad_z_tmp - grad_z) / np.linalg.norm(z_tmp - z),
                     np.linalg.norm(grad_z_tmp - grad_z_new) / np.linalg.norm(z_tmp - z_new))

        # stopping criterion: gradient norm
        xi = - (z_new - z) / self.stepsize - grad_z_tmp
        grad_xi = grad_z_new + xi
        self.grad_xi = grad_xi
        # print(np.linalg.norm(grad_xi), self.i_iter)
        # if np.linalg.norm(grad_xi) < 1e-6:
        #     self.stop = True

        # update decision variables
        self.x = x_new
        self.lg = lg_new
        self.grad = [grad_x_new, grad_lg_new]

        # update the objective
        return obj_val

    def record_stats(self):
        # inherited by other scaling relaxations using lg
        super().record_stats()
        self.stats.lg_trajectory.append(copy.deepcopy(self.lg))

    def stopping_criteria(self):
        # inherited by other scaling relaxations with same stopping criterion
        return self.stop


class ScaledGammaComplExtragradientAlgorithm(ScaledGammaExtragradientAlgorithm):

    def x_to_X_2(self, x, lg):
        xg = (1 - x) * np.exp(lg)
        return self.aux.W.T @ np.diag(xg) @ self.aux.W

    def grad_z(self, x, lg):
        # gradient wrt (x, lg)
        obj_val, Y = self.obj_n_grad_core(self.x_to_X_2(x, lg), self.data.d - self.data.s)
        grad_xg = np.diag(self.aux.W @ Y @ self.aux.W.T)
        g = np.exp(lg)
        grad_x = -grad_xg * g - lg
        grad_lg = grad_xg * (1 - x) * g + (1 - x)
        obj_val += (1 - x) @ lg
        return grad_x, grad_lg, obj_val - self.aux.log_det_C

