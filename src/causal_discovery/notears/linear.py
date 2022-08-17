import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, num_vars, max_iter=1000, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
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
    def _loss(U,V):
        """Evaluate value and gradient of loss."""
       # U, V = W[:num_vars, :num_vars], W[num_vars:, :num_vars]  # TODO: W not necessarily squared (V not necessarily squared)
        M = X_in @ U
        if loss_type == 'l2':
            R = X_in - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X_in.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X_in * M).sum()
            G_loss = 1.0 / X.shape[0] * X_in.T @ (sigmoid(M) - X_in)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X_in * M).sum()
            G_loss = 1.0 / X.shape[0] * X_in.T @ (S - X_in)
        elif loss_type =='l2c':
            M = (X_in * sigmoid(C @ V)) @ U # TODO: check order
            R = X_in - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
         #   G_loss = - 1.0 / X.shape[0] * X.T @ R
            G_loss_U = - 1.0 / X.shape[0] * (X_in * sigmoid(C @ V)).T @ R
            G_loss_V = - 1.0 / X.shape[0] * np.einsum('kl,l->kl', C.T @ (R * X_in * _sigmoid_der(C @ V)), U @ np.ones(num_vars))
         #   G_loss = np.concatenate((G_loss_U, G_loss_V), axis=1)
           # G_loss[:num_vars,:num_vars] = G_loss_U
           # G_loss[num_vars:,:num_vars] = G_loss_V
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss_U, G_loss_V

    def _sigmoid_der(X):
        return sigmoid(X) * (1 - sigmoid(X))

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
     #   U, V = W[:num_vars, :num_vars], W[num_vars:, :num_vars]
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + U * U / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w, dim1, dim2):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:dim1 * dim2] - w[dim1 * dim2:]).reshape([dim1, dim2])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        u,v = w[:2 * num_vars * num_vars], w[2 * num_vars * num_vars:]
        U,V = _adj(u,dim1=num_vars,dim2=num_vars), v.reshape([d-num_vars,num_vars])
        loss, G_loss_U, G_loss_V = _loss(U,V)
        h, G_h = _h(U)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * u.sum()
        temp = np.zeros_like(G_loss_U)
        temp[:num_vars, :num_vars] = (rho * h + alpha) * G_h
        G_smooth_U = G_loss_U + temp
        g_obj = np.concatenate((np.concatenate((G_smooth_U + lambda1, - G_smooth_U + lambda1), axis=None), G_loss_V.flatten()))
        return obj, g_obj

    X_in, C = X[:, :num_vars], X[:, num_vars:]
    n, d = X.shape # TODO: add dimension
   # d1, d2 = num_vars, d - num_vars

    u_est, rho, alpha, h = np.random.normal(loc=0.0, scale=0.1, size=(2*num_vars*num_vars)), 1.0, 0.0, np.inf
    v_est = np.random.normal(loc=1.0, size=((d-num_vars) * num_vars))
   # w_est, rho, alpha, h = np.random.normal(size=(2 * d * d)), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    w_est = np.concatenate((u_est,v_est))
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(num_vars) for j in range(num_vars)]
    bnds.extend([(None, None) for i in range(d-num_vars) for j in range(num_vars)])
    if loss_type == 'l2':
        X_in = X_in - np.mean(X_in, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new[:2*num_vars*num_vars],dim1=num_vars,dim2=num_vars))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
   # W_est = _adj(w_est)
    U_est = _adj(w_new[:2*num_vars*num_vars],dim1=num_vars,dim2=num_vars)
    V_est = w_new[2*num_vars*num_vars:].reshape([d-num_vars,num_vars])
    U_est[np.abs(U_est) < w_threshold] = 0
   # V_est[np.abs(V_est) < w_threshold] = 0
    return U_est, V_est


if __name__ == '__main__':
    from notears import utils
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

