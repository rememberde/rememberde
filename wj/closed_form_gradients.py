"""
Closed-form gradient verification for Methods 1 and 2.
Compares hand-derived gradients against PyTorch autograd.

==================================================================
Method 1: Free energy  F = E(Z,A) - T * S(Q)
==================================================================
  E = sum_{i,j} BCE(sigmoid(z_i . z_j), A_ij)     (plain, unweighted)
  S = (1/N) sum_i H(q_i),   q_ik = softmax(W z_i)_k

  Closed-form gradients (derived in the writeup):

    dE/dz_i = 2 * sum_j (sigma(z_i.z_j) - A_ij) * z_j           (symmetric A)

    dS/dz_i = -(1/N) [ a_i - q_bar_i * (b_i + 1) ]
      where
        a_i     = sum_k q_ik * w_k * (log q_ik + 1)      in R^d
        b_i     = sum_k q_ik * log q_ik  =  -H_i          scalar
        q_bar_i = sum_k q_ik * w_k                         in R^d

    dS/dw_k = -(1/N) sum_i q_ik * (log q_ik + H_i) * z_i

    dF/dz_i = dE/dz_i - T * dS/dz_i
    dF/dw_k =             - T * dS/dw_k      (E independent of W)

==================================================================
Method 2: Microstate counting  ln W  (Stirling, soft binning)
==================================================================
  Bin centers  c_1, ..., c_M  in R^d
  Soft bin assignment:
      w_ib = softmax_b( -||z_i - c_b||^2 / (2 sigma^2) )
  Soft community assignment:
      q_ik = softmax_k( W z_i )_k
  Soft counts:
      n_kb = sum_i q_ik * w_ib          n_k = sum_b n_kb
      p_kb = n_kb / n_k
  Stirling (leading order):
      ln W = -sum_{k,b} n_kb * log p_kb

  Partial derivative (key identity):
      d(ln W) / d n_kb = - log p_kb

  Chain rule:
      dn_kb/dz_i = q_ik * w_ib * [ (c_b - mu_bin_i)/sigma^2 + (w_k - q_bar_i) ]
        where  mu_bin_i = sum_b w_ib * c_b,   q_bar_i = sum_k q_ik * w_k

  Closed form (factored via node-specific aggregates):
      d(ln W)/dz_i = -L_i * [ (c_hat_i - mu_bin_i)/sigma^2 + (w_hat_i - q_bar_i) ]
    where
      L_i      = sum_{k,b} q_ik * w_ib * log p_kb              scalar (<= 0)
      c_hat_i  = (1/L_i) * sum_{k,b} q_ik * w_ib * log p_kb * c_b
      w_hat_i  = (1/L_i) * sum_{k,b} q_ik * w_ib * log p_kb * w_k

  Interpretation:
    - (c_hat_i - mu_bin_i): direction toward "high log p" bins
    - (w_hat_i - q_bar_i):  direction toward "high log p" community prototypes
    - -L_i >= 0 scales the step; gradient ascent on ln W reinforces the macrostate.
"""

import torch
import torch.nn.functional as F


# ============================== Method 1 ==============================
def method1_forward(Z, W, A, T):
    """Returns (F, E, S, Q). All differentiable."""
    S_logit = Z @ Z.t()
    E = F.binary_cross_entropy_with_logits(S_logit, A, reduction='sum')
    logits = Z @ W.t()
    Q = F.softmax(logits, dim=-1)
    H_i = -(Q * torch.log(Q + 1e-12)).sum(dim=-1)
    S = H_i.mean()
    return E - T * S, E, S, Q


def method1_grad_Z(Z, W, A, T):
    """Closed-form dF/dZ, shape (N, d)."""
    N = Z.shape[0]
    Sig = torch.sigmoid(Z @ Z.t())
    dE_dZ = 2.0 * (Sig - A) @ Z                 # (N, d)

    Q = F.softmax(Z @ W.t(), dim=-1)            # (N, K)
    logQ = torch.log(Q + 1e-12)
    q_bar = Q @ W                               # (N, d)
    a = (Q * (logQ + 1.0)) @ W                  # (N, d)
    b = (Q * logQ).sum(dim=-1, keepdim=True)    # (N, 1)
    dS_dZ = -(1.0 / N) * (a - q_bar * (b + 1.0))
    return dE_dZ - T * dS_dZ


def method1_grad_W(Z, W, A, T):
    """Closed-form dF/dW, shape (K, d)."""
    N = Z.shape[0]
    Q = F.softmax(Z @ W.t(), dim=-1)
    logQ = torch.log(Q + 1e-12)
    H_i = -(Q * logQ).sum(dim=-1, keepdim=True)     # (N, 1)
    coef = Q * (logQ + H_i)                         # (N, K)
    dS_dW = -(1.0 / N) * (coef.t() @ Z)             # (K, d)
    return -T * dS_dW


# ============================== Method 2 ==============================
def method2_forward(Z, W, C, sigma, eps=1e-12):
    """Returns ln W (Stirling, soft binning). Differentiable."""
    Z_sq = (Z ** 2).sum(dim=1, keepdim=True)        # (N, 1)
    C_sq = (C ** 2).sum(dim=1, keepdim=True).t()    # (1, M)
    dist2 = Z_sq - 2.0 * Z @ C.t() + C_sq           # (N, M)
    Wb = F.softmax(-dist2 / (2.0 * sigma ** 2), dim=-1)   # (N, M)
    Q = F.softmax(Z @ W.t(), dim=-1)                # (N, K)
    Nk = Q.t() @ Wb                                 # (K, M)  soft counts
    n_k = Nk.sum(dim=1, keepdim=True)               # (K, 1)
    p_kb = Nk / (n_k + eps)                         # (K, M)
    lnW = -(Nk * torch.log(p_kb + eps)).sum()
    return lnW


def method2_grad_Z(Z, W, C, sigma, eps=1e-12):
    """Closed-form d(ln W)/dZ, shape (N, d)."""
    Z_sq = (Z ** 2).sum(dim=1, keepdim=True)
    C_sq = (C ** 2).sum(dim=1, keepdim=True).t()
    dist2 = Z_sq - 2.0 * Z @ C.t() + C_sq
    Wb = F.softmax(-dist2 / (2.0 * sigma ** 2), dim=-1)   # (N, M)
    Q = F.softmax(Z @ W.t(), dim=-1)                # (N, K)
    Nk = Q.t() @ Wb                                 # (K, M)
    n_k = Nk.sum(dim=1, keepdim=True)
    p_kb = Nk / (n_k + eps)
    log_p = torch.log(p_kb + eps)                   # (K, M)

    mu_bin = Wb @ C                                 # (N, d)
    q_bar = Q @ W                                   # (N, d)

    WL = Wb @ log_p.t()                             # (N, K): WL[i,k] = sum_b w_ib log p_kb
    L = (Q * WL).sum(dim=-1, keepdim=True)          # (N, 1)

    Qlogp = Q @ log_p                               # (N, M): Qlogp[i,b] = sum_k q_ik log p_kb
    c_hat_num = (Qlogp * Wb) @ C                    # (N, d)
    c_hat = c_hat_num / (L.abs() + eps) * torch.sign(L + eps)
    # safer: c_hat = c_hat_num / L  when L != 0
    c_hat = c_hat_num / (L + eps * (L.abs() < eps).float())

    w_hat_num = (Q * WL) @ W                        # (N, d)
    w_hat = w_hat_num / (L + eps * (L.abs() < eps).float())

    grad = -L * ((c_hat - mu_bin) / sigma ** 2 + (w_hat - q_bar))
    return grad


# ============================== Verification ==============================
def _to_double(t):
    return t.detach().clone().double()


def check_method1(N=10, d=4, K=3, T=0.7, seed=0):
    torch.manual_seed(seed)
    Z = torch.randn(N, d, dtype=torch.float64, requires_grad=True)
    W = torch.randn(K, d, dtype=torch.float64, requires_grad=True)
    A = (torch.rand(N, N, dtype=torch.float64) > 0.5).double()
    A = (A + A.t()).clamp(max=1.0)            # symmetric, in {0,1}

    F_val, _, _, _ = method1_forward(Z, W, A, T)
    F_val.backward()
    auto_Z = Z.grad.clone()
    auto_W = W.grad.clone()

    closed_Z = method1_grad_Z(Z.detach(), W.detach(), A, T)
    closed_W = method1_grad_W(Z.detach(), W.detach(), A, T)

    err_Z = (auto_Z - closed_Z).abs().max().item()
    err_W = (auto_W - closed_W).abs().max().item()
    print(f"[Method 1]  max|auto - closed|   dZ: {err_Z:.2e}   dW: {err_W:.2e}")
    return err_Z, err_W


def check_method2(N=10, d=4, K=3, M=6, sigma=0.6, seed=0):
    torch.manual_seed(seed)
    Z = torch.randn(N, d, dtype=torch.float64, requires_grad=True)
    W = torch.randn(K, d, dtype=torch.float64)
    C = torch.randn(M, d, dtype=torch.float64)

    lnW = method2_forward(Z, W, C, sigma)
    lnW.backward()
    auto_Z = Z.grad.clone()

    closed_Z = method2_grad_Z(Z.detach(), W, C, sigma)

    err_Z = (auto_Z - closed_Z).abs().max().item()
    rel_Z = err_Z / (auto_Z.abs().max().item() + 1e-12)
    print(f"[Method 2]  max|auto - closed|   dZ: {err_Z:.2e}   rel: {rel_Z:.2e}")
    return err_Z


def check_multiple_seeds():
    print("Verifying closed-form gradients against PyTorch autograd (float64) ...\n")
    max_e1 = 0.0
    max_e2 = 0.0
    for s in range(5):
        e1z, e1w = check_method1(seed=s)
        e2 = check_method2(seed=s)
        max_e1 = max(max_e1, e1z, e1w)
        max_e2 = max(max_e2, e2)
    print()
    print(f"Max error across 5 seeds:  Method 1 = {max_e1:.2e}   Method 2 = {max_e2:.2e}")
    tol = 1e-6
    if max_e1 < tol and max_e2 < tol:
        print(f"PASS  (all errors < {tol})  -- closed-form derivations verified.")
    else:
        print(f"FAIL  (tolerance {tol})  -- inspect the formulas.")


if __name__ == "__main__":
    check_multiple_seeds()
