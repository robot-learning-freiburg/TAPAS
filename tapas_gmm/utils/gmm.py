import numpy as np

# from pbdlib.mvn import MVN as MVN_PBD
from riepybdlib.statistics import GMM as GMM_RBD
from riepybdlib.statistics import HMM as HMM_RBD
from riepybdlib.statistics import Gaussian as MVN_RBD
from scipy.linalg import block_diag, cho_factor, cho_solve

from tapas_gmm.utils.misc import multiply_iterable

# def concat_mvn(gaussians):
#     mvn = MVN_PBD()
#     mvn.mu = np.concatenate([g.mu for g in gaussians])
#     mvn._sigma = block_diag(*[g.sigma for g in gaussians])
#     mvn._lmbda = block_diag(*[g.lmbda for g in gaussians])

#     return mvn


def concat_mvn_rbd(gaussians):
    raise NotImplementedError
    manis = [g.manifold for g in gaussians]
    joint_manifold = multiply_iterable(manis)
    print(type(gaussians[0].mu), type(gaussians[0].sigma))
    joint_mu = np.concatenate([g.mu for g in gaussians])
    mvn = MVN_RBD(joint_manifold, joint_mu, joint_sigma)


# Adapted from https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df?permalink_comment_id=4396301#gistcomment-4396301
def kl_mvn(to: MVN_RBD, fr: MVN_RBD) -> float:
    m_to, S_to = to.get_mu_sigma(as_np=True)
    m_fr, S_fr = fr.get_mu_sigma(as_np=True)

    raise NotImplementedError("Not implemented for Riemannian.")

    d = m_fr - m_to

    c, lower = cho_factor(S_fr)

    def solve(B):
        return cho_solve((c, lower), B)

    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)

    return (term1 + term2 + term3 - len(d)) / 2.0


def kl_monte_carlo(to: MVN_RBD, fr: MVN_RBD, n_samples: int = int(1e5)) -> float:
    """
    Monte Carlo approximation of the KL divergence between two MVNs.
    """
    samples = to.sample(n_samples)
    return np.mean(to.prob(samples, log=True) - fr.prob(samples, log=True))


def hmm_transition_probabilities(
    first: HMM_RBD,
    second: HMM_RBD,
    first_idcs: list[int] | None = None,
    second_idcs: list[int] | None = None,
    drop_action_dim: bool = False,
    drop_rotation_dim: bool = False,
    includes_time: bool | None = None,
    sigma_scale: float | None = None,
    models_are_sequential: bool = False,
) -> np.ndarray:
    """
    Calculate the transition probabilities between two HMMs.
    Assumes that the start position of first and the end position of second are both
    forced to a fixed component.
    """
    if models_are_sequential:
        first_gaussians = [first.gaussians[-1].copy()]
        second_gaussians = [second.gaussians[0].copy()]
    else:
        first_gaussians = [g.copy() for g in first.gaussians]
        second_gaussians = [g.copy() for g in second.gaussians]

    if drop_action_dim or drop_rotation_dim:
        assert includes_time is not None, "Must specify whether model includes time."
        start = int(includes_time)
        stop = start + int(not drop_action_dim) + int(not drop_rotation_dim) + 1
        step = 1 + int(drop_rotation_dim and not drop_action_dim)

        first_idcs = first_idcs[start:stop:step]
        second_idcs = second_idcs[start:stop:step]

    kld = np.zeros((len(first_gaussians), len(second_gaussians)))

    if first_idcs is not None:
        for i, first_gauss in enumerate(first_gaussians):
            first_gaussians[i] = first_gauss.margin(first_idcs)

    if sigma_scale is not None:
        for i, first_gauss in enumerate(first_gaussians):
            first_gaussians[i] = MVN_RBD(
                first_gauss.manifold,
                first_gauss.mu,
                first_gauss.sigma * sigma_scale,
            )

    if second_idcs is not None:
        for i, second_gauss in enumerate(second_gaussians):
            second_gaussians[i] = second_gauss.margin(second_idcs)

    if sigma_scale is not None:
        for i, second_gauss in enumerate(second_gaussians):
            second_gaussians[i] = MVN_RBD(
                second_gauss.manifold,
                second_gauss.mu,
                second_gauss.sigma * sigma_scale,
            )

    for i, first_gauss in enumerate(first_gaussians):
        for j, second_gauss in enumerate(second_gaussians):
            assert first_gauss.manifold.name == second_gauss.manifold.name, (
                "Transition probabilities only defined for HMMs with same manifold. Forgot to"
                " marginalize to common frames?"
            )

            kld[i, j] = kl_monte_carlo(first_gauss, second_gauss)

    return np.exp(-kld)


def get_component_mu_sigma(
    model,
    start_idx,
    stop_idx,
    time_based: bool,
    xdx_based: bool,
    mu_on_tangent: bool = True,
):
    if time_based:
        raise NotImplementedError
        # TODO: for k in range(start, stop) and stack?
        pos_mu, pos_sigma = model.get_mu_sigma(idx=[0, k], stack=True, as_np=True)
    elif xdx_based:
        raise NotImplementedError
        k = j * n_rows * 6 + r * 3 + d + (1 if model_includes_time else 0)
        # print(j, m, r, k, k+ 3 * n_rows)
        # print(model.mu.shape)
        if rot_on_tangent:
            mu, sigma = model.get_mu_sigma(
                idx=[k, k + 3 * n_rows], stack=True, as_np=True
            )
    else:
        mu, sigma = model.get_mu_sigma(
            idx=list(range(start_idx, stop_idx)),
            as_np=True,
            stack=True,
            mu_on_tangent=mu_on_tangent,
        )

    return mu, sigma
