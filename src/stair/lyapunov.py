"""
Lyapunov spectrum for a discrete-time map  x_{l+1} = F_l(x_l), l=0..L-1.

We treat the full flattened state (B=1, T, D) -> R^{T*D} as the phase space
and compute the spectrum of the product   J = J_{L-1} ... J_1 J_0
where J_l is the layer Jacobian at the trajectory point.

For numerical stability we use the QR method:
  Q_0 = I (n x k, k <= n)   # k = number of exponents to track
  for l in 0..L-1:
      M_l = J_l @ Q_l
      Q_{l+1}, R_l = qr(M_l)
      accumulate log|diag(R_l)|
  lambda_i = sum_l log|R_l[i,i]| / L

This gives the *finite-time / local* Lyapunov spectrum for depth=L.

We compute J_l @ Q_l column-by-column via JVPs (forward-mode AD) so we never
materialize the full Jacobian.
"""
from __future__ import annotations
import torch
import numpy as np


@torch.no_grad()
def _qr(M):
    # column-pivoting QR not needed; use reduced QR
    Q, R = torch.linalg.qr(M, mode="reduced")
    # enforce positive diagonal of R (sign convention)
    sgn = torch.sign(torch.diagonal(R))
    sgn[sgn == 0] = 1.0
    Q = Q * sgn.unsqueeze(0)
    R = R * sgn.unsqueeze(1)
    return Q, R


def layer_jacobian_action(block, x_flat, V, input_shape):
    """
    Compute J_l @ V where J_l is the Jacobian of block at x_flat,
    V has shape (n, k). Uses forward-mode JVP per column to avoid materializing J.

    Returns: tensor (n, k)
    """
    n, k = V.shape
    cols = []
    for j in range(k):
        v = V[:, j].reshape(input_shape)

        def f(xf):
            return block(xf.reshape(input_shape)).reshape(-1)

        # jvp: (output, jvp_value)
        _, jvp_val = torch.func.jvp(f, (x_flat,), (v.reshape(-1),))
        cols.append(jvp_val)
    return torch.stack(cols, dim=1)


def lyapunov_spectrum(model, x0, k=None):
    """
    model: TinyTransformer
    x0   : input tensor (B=1, T, D)
    k    : number of exponents to track (default = full = T*D)

    Returns:
      lambdas : (k,) numpy array, sorted descending
      log_grow_history : (L, k) np array of cumulative log growth per layer
    """
    model.eval()
    x = x0.detach().clone()
    input_shape = x.shape
    n = x.numel()
    if k is None:
        k = n
    k = min(k, n)

    # initial orthonormal frame
    Q = torch.linalg.qr(torch.randn(n, k, dtype=x.dtype, device=x.device))[0]

    log_sum = torch.zeros(k, dtype=x.dtype, device=x.device)
    history = []

    x_flat = x.reshape(-1)
    for blk in model.blocks:
        # action of J_l on Q
        JQ = layer_jacobian_action(blk, x_flat, Q, input_shape)
        Q, R = _qr(JQ)
        diag = torch.diagonal(R).abs().clamp_min(1e-30)
        log_sum = log_sum + torch.log(diag)
        history.append(log_sum.detach().cpu().numpy().copy())
        # advance trajectory
        with torch.no_grad():
            x_next = blk(x_flat.reshape(input_shape))
        x_flat = x_next.reshape(-1).detach()

    L = len(model.blocks)
    lambdas = (log_sum / L).detach().cpu().numpy()
    lambdas_sorted = np.sort(lambdas)[::-1].copy()
    return lambdas_sorted, np.stack(history, axis=0)
