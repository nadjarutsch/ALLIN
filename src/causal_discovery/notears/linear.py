import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import scipy.stats as st


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est



def notears_linear_adv(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, thresh=0.01):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_abs = np.abs(W_est)
    Z = np.empty_like(W_est)
    Vars = 1 / (X.shape[0] - X.shape[1] - 1) * np.sum(((X - X @ W_est) ** 2), axis=0)

    for i in range(W_est.shape[0]):
        for j in range(W_est.shape[1]):
            if i == j:
                continue
            X_temp = X.copy()
            X_temp = np.delete(X_temp, j, axis=1)
            idx = i if i < j else i-1
            Z[i, j] = W_abs[i, j] / (np.sqrt(Vars[j]) * np.sqrt((np.linalg.inv(X_temp.T @ X_temp)).diagonal()[idx]))

    Probs = st.norm.cdf(Z)
    W_est[Probs < 1 - thresh] = 0
   # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est




def allin_linear(X, P, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        W_obs_augmented, W_int_augmented = np.split(W, 2, axis=0)
        X_augmented = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)   # TODO: doublecheck
        M_obs = X_augmented @ W_obs_augmented
        M_int = X_augmented @ W_int_augmented
    #    M = X @ W
        if loss_type == 'l2':
            R_obs = X - M_obs
            R_int = X - M_int
         #   R = X - M
            loss = 0.5 / X.shape[0] * ((P * (R_obs ** 2)).sum() + ((1 - P) * (R_int ** 2)).sum())
            G_loss_obs = - 1.0 / X.shape[0] * (X_augmented.T @ (P * R_obs))
            G_loss_int = - 1.0 / X.shape[0] * (X_augmented.T @ ((1 - P) * R_int))
            G_loss_int[:-1, ...] = 0
            G_loss = np.concatenate((G_loss_obs, G_loss_int), axis=0)
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        W_obs_augmented, W_int_augmented = np.split(W, 2)
        W_obs = W_obs_augmented[:-1, ...]    # TODO: doublecheck
        E = slin.expm(W_obs * W_obs)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W_obs * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        w_obs = w[:2 * d * d]
        w_obs_bias = w[2 * d * d:2 * d * d + 2 * d]
        w_int = w[2 * d * d + 2 * d:4 * d * d + 2 * d]
        w_int_bias = w[4 * d * d + 2 * d: 4 * d * d + 4 * d]

        W_obs = (w_obs[:d * d] - w_obs[d * d:]).reshape([d, d])
        W_obs_bias = (w_obs_bias[:d] - w_obs_bias[d:]).reshape([1, d])
        W_int = (w_int[:d * d] - w_int[d * d:]).reshape([d, d])
        W_int_bias = (w_int_bias[:d] - w_int_bias[d:]).reshape([1, d])

        return np.concatenate((W_obs, W_obs_bias, W_int, W_int_bias), axis=0)

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w[:2 * d * d].sum()
     #   G_h = np.concatenate((G_h, np.zeros((d + 2, d))), axis=0)
        G_smooth = G_loss[:d, ...] + (rho * h + alpha) * G_h
        g_obj_obs = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        g_obj_bias_obs = np.concatenate((G_loss[d:d+1, ...], - G_loss[d:d+1, ...]), axis=None)
        g_obj_int = np.concatenate((G_loss[d+1:2*d+1, ...], - G_loss[d+1:2*d+1, ...]), axis=None)
        g_obj_bias_int = np.concatenate((G_loss[-1, ...], - G_loss[-1, ...]), axis=None)
        g_obj = np.concatenate((g_obj_obs, g_obj_bias_obs, g_obj_int, g_obj_bias_int), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est_obs, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    w_est_obs_augm = np.concatenate((w_est_obs, np.zeros(2 * d)))
    w_est_int = np.zeros(2 * d * d)
    w_est_int_augm = np.concatenate((w_est_int, np.zeros(2 * d)))
    bnds_obs = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    bnds_obs_bias = [(0, None) for _ in range(2) for i in range(d)]
    bnds_int = [(0, 0) for _ in range(2) for i in range(d) for j in range(d)]
    bnds_int_bias = [(0, None) for _ in range(2) for i in range(d)]

    w_est = np.concatenate((w_est_obs_augm, w_est_int_augm))
    bnds = bnds_obs + bnds_obs_bias + bnds_int + bnds_int_bias
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
   # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def allin_linear_adv(X, P, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, thresh=0.01):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        W_obs_augmented, W_int_augmented = np.split(W, 2, axis=0)
        X_augmented = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)   # TODO: doublecheck
        M_obs = X_augmented @ W_obs_augmented
        M_int = X_augmented @ W_int_augmented
    #    M = X @ W
        if loss_type == 'l2':
            R_obs = X - M_obs
            R_int = X - M_int
         #   R = X - M
            loss = 0.5 / X.shape[0] * ((P * (R_obs ** 2)).sum() + ((1 - P) * (R_int ** 2)).sum())
            G_loss_obs = - 1.0 / X.shape[0] * (X_augmented.T @ (P * R_obs))
            G_loss_int = - 1.0 / X.shape[0] * (X_augmented.T @ ((1 - P) * R_int))
            G_loss_int[:-1, ...] = 0
            G_loss = np.concatenate((G_loss_obs, G_loss_int), axis=0)
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        W_obs_augmented, W_int_augmented = np.split(W, 2)
        W_obs = W_obs_augmented[:-1, ...]    # TODO: doublecheck
        E = slin.expm(W_obs * W_obs)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W_obs * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        w_obs = w[:2 * d * d]
        w_obs_bias = w[2 * d * d:2 * d * d + 2 * d]
        w_int = w[2 * d * d + 2 * d:4 * d * d + 2 * d]
        w_int_bias = w[4 * d * d + 2 * d: 4 * d * d + 4 * d]

        W_obs = (w_obs[:d * d] - w_obs[d * d:]).reshape([d, d])
        W_obs_bias = (w_obs_bias[:d] - w_obs_bias[d:]).reshape([1, d])
        W_int = (w_int[:d * d] - w_int[d * d:]).reshape([d, d])
        W_int_bias = (w_int_bias[:d] - w_int_bias[d:]).reshape([1, d])

        return np.concatenate((W_obs, W_obs_bias, W_int, W_int_bias), axis=0)

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w[:2 * d * d].sum()
     #   G_h = np.concatenate((G_h, np.zeros((d + 2, d))), axis=0)
        G_smooth = G_loss[:d, ...] + (rho * h + alpha) * G_h
        g_obj_obs = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        g_obj_bias_obs = np.concatenate((G_loss[d:d+1, ...], - G_loss[d:d+1, ...]), axis=None)
        g_obj_int = np.concatenate((G_loss[d+1:2*d+1, ...], - G_loss[d+1:2*d+1, ...]), axis=None)
        g_obj_bias_int = np.concatenate((G_loss[-1, ...], - G_loss[-1, ...]), axis=None)
        g_obj = np.concatenate((g_obj_obs, g_obj_bias_obs, g_obj_int, g_obj_bias_int), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est_obs, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    w_est_obs_augm = np.concatenate((w_est_obs, np.zeros(2 * d)))
    w_est_int = np.zeros(2 * d * d)
    w_est_int_augm = np.concatenate((w_est_int, np.zeros(2 * d)))
    bnds_obs = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    bnds_obs_bias = [(0, None) for _ in range(2) for i in range(d)]
    bnds_int = [(0, 0) for _ in range(2) for i in range(d) for j in range(d)]
    bnds_int_bias = [(0, None) for _ in range(2) for i in range(d)]

    w_est = np.concatenate((w_est_obs_augm, w_est_int_augm))
    bnds = bnds_obs + bnds_obs_bias + bnds_int + bnds_int_bias
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
   # W_est[np.abs(W_est) < w_threshold] = 0
    W_obs_augmented, W_int_augmented = np.split(W_est, 2)
    W_obs = W_obs_augmented[:-1, ...]

    W_abs = np.abs(W_obs)
    Z = np.zeros_like(W_obs)
    X_augmented = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    Vars_obs = np.sum(P * (X - X_augmented @ W_obs_augmented) ** 2, axis=0)
    Vars_int = np.sum((1 - P) * (X - X_augmented @ W_int_augmented) ** 2, axis=0)
    Vars = 1 / (X.shape[0] - X.shape[1] - 1) * (Vars_obs + Vars_int)

    for i in range(W_obs.shape[0]):
        for j in range(W_obs.shape[1]):
            if i == j:
                continue
            X_temp = (P[..., j][:, None] * X).copy()
            X_temp = np.delete(X_temp, j, axis=1)
            idx = i if i < j else i - 1
            V_test = np.sqrt((np.linalg.inv(X.T @ X)).diagonal()[idx])
            V = np.sqrt((np.linalg.inv(X_temp.T @ X_temp)).diagonal()[idx])     # TODO: doublecheck
            Z[i, j] = W_abs[i, j] / (np.sqrt(Vars[j]) * V)

    Probs = st.norm.cdf(Z)
    W_obs[Probs < 1 - thresh] = 0
    W_est[:d, ...] = W_obs

    return W_est


if __name__ == '__main__':
    from causal_discovery.notears import utils
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

