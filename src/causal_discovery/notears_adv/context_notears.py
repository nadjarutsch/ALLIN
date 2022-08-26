import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def context_notears_linear(X,
                           lambda1,
                           num_vars,
                           max_iter=1000,
                           h_tol=1e-8,
                           rho_max=1e+16,
                           w_threshold=0.3):
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
    def _loss(W,V):
        """Evaluate value and gradient of loss."""

        #M = (X_in * sigmoid(C @ V)) @ U
        CV = np.tensordot(C, V, axes=([1,0]))
        M = np.einsum('nv,nvd->nd', X_in, sigmoid(CV) * W[None, :, :])
        #M = np.einsum('nv,nvd->nd', X_in, np.ones_like(sigmoid(CV)) * W[None, :, :])

        R = X_in - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        #G_loss_U = - 1.0 / X.shape[0] * (X_in * sigmoid(C @ V)).T @ R
        #G_loss_V = - 1.0 / X.shape[0] * C.T @ (X_in * sigmoid_der(C @ V) * (R @ U.T))
        #G_loss_V = - 1.0 / X.shape[0] * np.einsum('kl,dl->kl', C.T @ (R * X_in), U)
     #   G_loss = np.concatenate((G_loss_U, G_loss_V), axis=1)
        G_loss_W = - 1.0 / X.shape[0] * np.sum(R[:,None,:] * X_in[:,:,None] * sigmoid(CV), axis=0)

        #G_loss_W = - 1.0 / X.shape[0] * np.sum(R[:, None, :] * X_in[:, :, None] * np.ones_like(sigmoid(CV)), axis=0)
        '''
        M = X_in @ W

        R = X_in - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss_W = - 1.0 / X.shape[0] * X_in.T @ R
        '''

        return loss, G_loss_W

    def sigmoid_der(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - num_vars
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w, dim1, dim2):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:dim1 * dim2] - w[dim1 * dim2:]).reshape([dim1, dim2])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
      #  u,v = w[:2 * num_vars * num_vars], w[2 * num_vars * num_vars:]
      #  U,V = _adj(u, dim1=num_vars, dim2=num_vars), _adj(v, dim1=d-num_vars, dim2=num_vars) #v.reshape([d-num_vars, num_vars])
        W = _adj(w, dim1=num_vars, dim2=num_vars)
        loss, G_loss_W = _loss(W, V)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth_W = G_loss_W + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth_W + lambda1, - G_smooth_W + lambda1), axis=None)
        # G_V = np.concatenate((G_loss_V + lambda1, - G_loss_V + lambda1), axis=None)
        # G_V = G_loss_V.flatten()
        # g_obj = np.concatenate((G_U, G_V))
        return obj, g_obj

    X_in, C = X[:, :num_vars], X[:, num_vars:]
    n, d = X.shape
    V = np.ones((d - num_vars, num_vars, num_vars)) * 10 # all gates open for test
    V[list(range(1, num_vars+1)), ..., list(range(num_vars))] = -10
   # d1, d2 = num_vars, d - num_vars

    w_est, rho, alpha, h = np.zeros(2*num_vars*num_vars), 1.0, 0.0, np.inf
   # v_est = np.random.normal(loc=1.0, size=(2 * (d - num_vars) * num_vars))
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(num_vars) for j in range(num_vars)]
   # bnds.extend([(None, None) for _ in range(2) for i in range(d - num_vars) for j in range(num_vars)])

    X_in = X_in - np.mean(X_in, axis=0, keepdims=True) # normalize input (zero-center)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new, dim1=num_vars, dim2=num_vars))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est, dim1=num_vars, dim2=num_vars)
    # V_est = w_new[2*num_vars*num_vars:].reshape([d-num_vars,num_vars])
    #V_est = _adj(w_new[2 * num_vars * num_vars:], dim1=d-num_vars, dim2=num_vars)
    W_est[np.abs(W_est) < w_threshold] = 0

    return W_est

