from base import *


class OScaledLinxExtragradient(MESProblemAlgorithm):
    # linx scaled by a scalar factor, i.e., o-scaling

    def __init__(self):
        super().__init__()
        self.lg = 0  # log of scaling factor
        self.L = 1.  # initial value of local Lipschitz constant
        self.stepsize = .1  # initial value of stepsize
        self.grad = None  # [grad_x, grad_lg]
        self.eps = 1e-10  # tolerance to prevent division by 0
        self.grad_xi = None  # a measurement of convergence / saddle point gap
        self.stop = False  # whether stopping criterion is satisfied

    def initialize(self, initial_pt=None):
        super().initialize(initial_pt)
        self.val = self.obj_slinx(self.x, self.lg)  # redefine initial objective
        self.stats.primal_val = [self.val]  # re-record objective
        self.stats.lg_trajectory.append(self.lg)  # record lg (log of scaling factor) in addition to x
        self.grad = self.grad_z(self.x, self.lg)  # compute initial gradient

    def grad_lg(self, x, lg, L_inv):
        # gradient wrt the log of scaling factor
        # - diag(L^{-1}) @ (x-1) - d + s
        # Theorem 3.3.13 in [Fampa and Lee, 2022] (page 58); see Section 3.3.5 (page 57) for definition of L
        # it is the gradient of log(factor), not the scaling factor itself
        return np.diag(L_inv) @ (1 - x) - self.data.d + self.data.s

    def grad_x(self, x, lg, L_inv):
        # gradient wrt x
        # -.5 * (gamma * diag(C @ L^{-1} @ C) - diag(L^{-1}))
        # Lemme 3.3.11 in [Fampa and Lee, 2022] (page 54)
        # see Section 3.3.5 (page 57) for definition of L
        # see also our notes for more general scaling reduced to a special case
        return -.5 * (np.exp(lg) * np.diag(self.data.C @ L_inv @ self.data.C) - np.diag(L_inv))

    def grad_z(self, x, lg):
        # gradient wrt (x, lg)
        L_inv = np.linalg.inv(self.matrix_L(x, lg))
        return self.grad_x(x, lg, L_inv), self.grad_lg(x, lg, L_inv)

    def obj_slinx(self, x, lg):
        # overwrite objective function in base class by the scaled version
        # -.5 * (logdet(L(x, lg)) - s * lg)
        # see definition from [Fampa and Lee, 2022] (page 54)
        return -.5 * (np.linalg.slogdet(self.matrix_L(x, lg))[1] - self.data.s * lg)

    def matrix_L(self, x, lg):
        # L(x, lg) = exp(lg) * C @ Diag(x) @ C + Diag(1-x)
        # see definition from [Fampa and Lee, 2022] (page 54)
        return np.exp(lg) * (self.data.C @ np.diag(x) @ self.data.C) + np.diag(1 - x)

    def stepsize_update(self):
        # inherited by other scaling relaxations
        theta = .5
        kappa = .75
        if self.i_iter >= 100:
            # self.stepsize = min(self.stepsize, theta / self.L)
            self.stepsize *= kappa
        elif self.i_iter >= 0:
            # initial phase: not enforcing monotonicity
            self.stepsize = theta / self.L

    def update(self):

        first_round = True  # enter the loop at least once
        while first_round or self.L * self.stepsize >= 1:

            if not first_round or self.i_iter < 100:
                # update stepsize
                self.stepsize_update()
            else:
                first_round = False

            # 1st gradient descent in x
            x_tmp = self.proj_x(self.x - self.stepsize * self.grad[0])
            # 1st gradient ascent in log scaling factor
            lg_tmp = self.lg + self.stepsize * self.grad[1]

            # 2nd gradient descent in x & ascent in log scaling factor
            grad_x_tmp, grad_lg_tmp = self.grad_z(x_tmp, lg_tmp)
            x_new = self.proj_x(self.x - self.stepsize * grad_x_tmp)
            lg_new = self.lg + self.stepsize * grad_lg_tmp

            # update local Lipschitz constant
            # stack the primal and dual variables into one vector
            z = np.append(self.x, self.lg)
            z_tmp = np.append(x_tmp, lg_tmp)
            z_new = np.append(x_new, lg_new)
            # stack the primal and dual gradients into one vector
            grad_z = np.append(self.grad[0], self.grad[1])
            grad_z_tmp = np.append(grad_x_tmp, grad_lg_tmp)
            grad_x_new, grad_lg_new = self.grad_z(x_new, lg_new)
            grad_z_new = np.append(grad_x_new, grad_lg_new)
            # stop if the iterates are too close to compute the local Lipschitz constants
            if np.linalg.norm(z_tmp - z) < self.eps:
                self.stop = True
                return self.stats.primal_val[-1]
            # update local Lipschitz constant
            self.L = max(np.linalg.norm(grad_z_tmp - grad_z) / np.linalg.norm(z_tmp - z),
                         np.linalg.norm(grad_z_tmp - grad_z_new) / np.linalg.norm(z_tmp - z_new)
                         if np.linalg.norm(z_tmp - z_new) >= self.eps else 0)
            # no backtracking during initial stage
            if self.i_iter < 100:
                break

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

        # update objective value
        obj_new = self.obj_slinx(self.x, self.lg)
        return obj_new

    def record_stats(self):
        # inherited by other scaling relaxations using lg
        super().record_stats()
        self.stats.lg_trajectory.append(copy.deepcopy(self.lg))

    def stopping_criteria(self):
        # inherited by other scaling relaxations with same stopping criterion
        return self.stop


class GScaledLinxExtragradient(OScaledLinxExtragradient):

    def initialize(self, initial_pt=None):
        super(OScaledLinxExtragradient, self).initialize(initial_pt)
        self.lg = np.zeros(self.data.d)  # initialize log of scaling vector
        self.stats.lg_trajectory = [self.lg]  # record lg
        self.val = self.obj_slinx(self.x, self.lg)  # redefine initial objective
        self.stats.primal_val = [self.val]  # re-record objective
        self.grad = self.grad_z(self.x, self.lg)  # compute initial gradient

    def grad_x(self, x, lg, L_inv):
        # gradient wrt x
        # -.5 * diag(C @ Diag(g) @ L^{-1} @ Diag(g) @ C) - diag(L^{-1})) + log(g)
        # see our notes
        Dg = np.diag(np.exp(lg))
        return -.5 * (np.diag(self.data.C @ Dg @ L_inv @ Dg @ self.data.C) - np.diag(L_inv)) + lg

    def grad_lg(self, x, lg, L_inv):
        # gradient wrt the log of scaling vector
        # diag(L^{-1}) \circ (1-x) - (1-x)
        # Theorem 2.iii in [Chen et al, 2024] (page 12)
        # it is the gradient of log(scaling vector), not the scaling vector itself
        return np.diag(L_inv) * (1 - x) - (1 - x)
        # g = np.exp(lg)
        # C = self.data.C
        # return - g * np.diag(L_inv @ np.diag(g) @ C @ np.diag(x) @ C) + x

    def grad_z(self, x, lg):
        # gradient wrt (x, lg)
        L_inv = np.linalg.inv(self.matrix_L(x, lg))
        return self.grad_x(x, lg, L_inv), self.grad_lg(x, lg, L_inv)

    def matrix_L(self, x, lg):
        # L(x, lg) = Diag(g) @ C @ Diag(x) @ C @ Diag(g) + Diag(1-x)
        # see definition in [Chen et al, 2024] (page 12)
        Dg = np.diag(np.exp(lg))
        return Dg @ self.data.C @ np.diag(x) @ self.data.C @ Dg + np.diag(1 - x)

    def obj_slinx(self, x, lg):
        # -.5 * (logdet(L(x, lg)) - x @ lg)
        # see definition in Section 3 [Chen et al, 2024] (page 10)
        return -.5 * np.linalg.slogdet(self.matrix_L(x, lg))[1] + x @ lg

    def update(self):

        first_round = True  # enter the loop at least once
        while first_round or self.L * self.stepsize >= 1:

            if not first_round or self.i_iter < 100:
                # update stepsize
                self.stepsize_update()
            else:
                first_round = False

            # 1st gradient update
            x_tmp = self.proj_x(self.x - self.stepsize * self.grad[0])
            lg_tmp = self.lg + self.stepsize * self.grad[1]

            # 2nd gradient update
            grad_x_tmp, grad_lg_tmp = self.grad_z(x_tmp, lg_tmp)
            x_new = self.proj_x(self.x - self.stepsize * grad_x_tmp)
            lg_new = self.lg + self.stepsize * grad_lg_tmp

            # update local Lipschitz constant
            # stack the primal and dual variables into one vector
            z = np.hstack([self.x, self.lg])
            z_tmp = np.hstack([x_tmp, lg_tmp])
            z_new = np.hstack([x_new, lg_new])
            # stack the primal and dual gradients into one vector
            grad_z = np.hstack([self.grad[0], self.grad[1]])
            grad_z_tmp = np.hstack([grad_x_tmp, grad_lg_tmp])
            grad_x_new, grad_lg_new = self.grad_z(x_new, lg_new)
            grad_z_new = np.hstack([grad_x_new, grad_lg_new])
            # stop if the iterates are too close to compute the local Lipschitz constants
            if np.linalg.norm(z_tmp - z) < self.eps:
                self.stop = True
                return self.stats.primal_val[-1]
            # update local Lipschitz constant
            self.L = max(np.linalg.norm(grad_z_tmp - grad_z) / np.linalg.norm(z_tmp - z),
                         np.linalg.norm(grad_z_tmp - grad_z_new) / np.linalg.norm(z_tmp - z_new)
                         if np.linalg.norm(z_tmp - z_new) >= self.eps else 0)
            # no backtracking during initial stage
            if self.i_iter < 100:
                break

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
        obj_new = self.obj_slinx(self.x, self.lg)
        return obj_new


class GGScaledLinxExtragradient(OScaledLinxExtragradient):

    def initialize(self, initial_pt=None):
        super(OScaledLinxExtragradient, self).initialize(initial_pt)
        self.lg = np.zeros(2 * self.data.d)  # initialize log of scaling vector
        self.stats.lg_trajectory = [self.lg]  # record lg
        self.val = self.obj_slinx(self.x, self.lg)  # redefine initial objective
        self.stats.primal_val = [self.val]  # re-record objective
        self.grad = self.grad_z(self.x, self.lg)  # compute initial gradient

    def grad_x(self, x, lg, L_inv):
        # gradient wrt x
        # -.5 * g0 \circ diag(C @ Diag(g1) @ L^{-1} @ Diag(g1) @ C) + .5 * diag(L^{-1}) + .5 lg0 + .5 * lg1
        # see our notes
        g0, g1, lg0, lg1 = self.split_lg(lg)
        return -.5 * (g0 * np.diag(self.data.C @ np.diag(g1) @ L_inv @ np.diag(g1) @ self.data.C) - np.diag(
            L_inv)) + .5 * lg0 + lg1

    def grad_lg(self, x, lg, L_inv):
        # gradient wrt the log of scaling vector
        # grad_lg0 = -.5 * g0 \circ x \circ diag(C @ Diag(g1) @ L^{-1} @ Diag(g1) @ C) + .5 * x
        # grad_lg1 = (1 - x) \circ diag(L^{-1}) + x - 1
        # see our notes
        g0, g1, _, _ = self.split_lg(lg)
        grad_lg0 = -.5 * g0 * x * np.diag(self.data.C @ np.diag(g1) @ L_inv @ np.diag(g1) @ self.data.C) + .5 * x
        # grad_lg0 *= 2
        # grad_lg0 = np.zeros(self.data.d)
        grad_lg1 = (1 - x) * np.diag(L_inv) - (1 - x)
        # grad_lg1 = np.zeros(self.data.d)
        return np.hstack([grad_lg0, grad_lg1])

    def grad_z(self, x, lg):
        # gradient wrt (x, lg)
        L_inv = np.linalg.inv(self.matrix_L(x, lg))
        return self.grad_x(x, lg, L_inv), self.grad_lg(x, lg, L_inv)

    def split_lg(self, lg):
        lg0 = lg[:self.data.d]
        g0 = np.exp(lg0)
        lg1 = lg[-self.data.d:]
        g1 = np.exp(lg1)
        return g0, g1, lg0, lg1

    def matrix_L(self, x, lg):
        g0, g1, _, _ = self.split_lg(lg)
        return np.diag(g1) @ self.data.C @ np.diag(x * g0) @ self.data.C @ np.diag(g1) + np.diag((1 - x))

    def obj_slinx(self, x, lg):
        _, _, lg0, lg1 = self.split_lg(lg)
        return -.5 * np.linalg.slogdet(self.matrix_L(x, lg))[1] + .5 * x @ lg0 + x @ lg1

    def update(self):

        first_round = True  # enter the loop at least once
        while first_round or self.L * self.stepsize >= 1:

            if not first_round or self.i_iter < 100:
                # update stepsize
                self.stepsize_update()
            else:
                first_round = False

            # 1st gradient update
            x_tmp = self.proj_x(self.x - self.stepsize * self.grad[0])
            lg_tmp = self.lg + self.stepsize * self.grad[1]

            # 2nd gradient update
            grad_x_tmp, grad_lg_tmp = self.grad_z(x_tmp, lg_tmp)
            x_new = self.proj_x(self.x - self.stepsize * grad_x_tmp)
            lg_new = self.lg + self.stepsize * grad_lg_tmp

            # update local Lipschitz constant
            # stack the primal and dual variables into one vector
            z = np.hstack([self.x, self.lg])
            z_tmp = np.hstack([x_tmp, lg_tmp])
            z_new = np.hstack([x_new, lg_new])
            # stack the primal and dual gradients into one vector
            grad_z = np.hstack([self.grad[0], self.grad[1]])
            grad_z_tmp = np.hstack([grad_x_tmp, grad_lg_tmp])
            grad_x_new, grad_lg_new = self.grad_z(x_new, lg_new)
            grad_z_new = np.hstack([grad_x_new, grad_lg_new])
            # stop if the iterates are too close to compute the local Lipschitz constants
            if np.linalg.norm(z_tmp - z) < self.eps:
                self.stop = True
                return self.stats.primal_val[-1]
            # update local Lipschitz constant
            self.L = max(np.linalg.norm(grad_z_tmp - grad_z) / np.linalg.norm(z_tmp - z),
                         np.linalg.norm(grad_z_tmp - grad_z_new) / np.linalg.norm(z_tmp - z_new)
                         if np.linalg.norm(z_tmp - z_new) >= self.eps else 0)
            # no backtracking during initial stage
            if self.i_iter < 100:
                break

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
        obj_new = self.obj_slinx(self.x, self.lg)
        return obj_new
