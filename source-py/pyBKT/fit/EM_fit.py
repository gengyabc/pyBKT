#########################################
# EM_fit.py                             #
# EM_fit                                #
#                                       #
# @author Anirudhan Badrinath           #
# @author Christian Garay               #
# Last edited: 20 March 2020            #
#########################################

import numpy as np
from time import time
from pyBKT.util import check_data
from pyBKT.fit import M_step
from multiprocessing import Pool, cpu_count

_NUMERIC_EPS = 1e-12


def _safe_divide(numerator, denominator):
    denom = np.asarray(denominator, dtype=float)
    safe = np.where(np.abs(denom) < _NUMERIC_EPS, _NUMERIC_EPS, denom)
    return np.asarray(numerator, dtype=float) / safe


def _normalize_two_state(values):
    norm = float(np.sum(values))
    if norm < _NUMERIC_EPS:
        return np.array([0.5, 0.5], dtype=float)
    return values / norm


def EM_fit(model, data, tol=0.005, maxiter=100, parallel=True, fixed=None):
    fixed = fixed or {}
    check_data.check_data(data)

    num_subparts = data["data"].shape[0]
    num_resources = len(model["learns"])

    trans_softcounts = np.zeros((num_resources, 2, 2))
    emission_softcounts = np.zeros((num_subparts, 2, 2))
    init_softcounts = np.zeros((2, 1))
    log_likelihoods = np.zeros((maxiter, 1))

    result = {}
    result["all_trans_softcounts"] = trans_softcounts
    result["all_emission_softcounts"] = emission_softcounts
    result["all_initial_softcounts"] = init_softcounts

    for i in range(maxiter):
        result = run(
            data,
            model,
            result["all_trans_softcounts"],
            result["all_emission_softcounts"],
            result["all_initial_softcounts"],
            1,
            parallel,
            fixed=fixed,
        )
        for j in range(num_resources):
            result["all_trans_softcounts"][j] = result["all_trans_softcounts"][j].transpose()
        for j in range(num_subparts):
            result["all_emission_softcounts"][j] = result["all_emission_softcounts"][j].transpose()

        log_likelihoods[i, 0] = float(result["total_loglike"])
        if i > 1 and abs(log_likelihoods[i, 0] - log_likelihoods[i - 1, 0]) <= tol:
            break

        model = M_step.run(
            model,
            result["all_trans_softcounts"],
            result["all_emission_softcounts"],
            result["all_initial_softcounts"],
            fixed=fixed,
        )

    return model, log_likelihoods[: i + 1]


def run(data, model, trans_softcounts, emission_softcounts, init_softcounts, num_outputs, parallel=True, fixed=None):
    fixed = fixed or {}
    alldata = data["data"]
    bigT, num_subparts = len(alldata[0]), len(alldata)
    allresources, starts, learns, forgets, guesses, slips, lengths = (
        data["resources"],
        data["starts"],
        model["learns"],
        model["forgets"],
        model["guesses"],
        model["slips"],
        data["lengths"],
    )

    prior, num_sequences, num_resources = model["prior"], len(starts), len(learns)
    normalizeLengths = False

    if "prior" in fixed:
        prior = fixed["prior"]
    initial_distn = np.empty((2,), dtype=float)
    initial_distn[0] = 1 - prior
    initial_distn[1] = prior

    if "learns" in fixed:
        learns = learns * (fixed["learns"] < 0) + fixed["learns"] * (fixed["learns"] >= 0)
    if "forgets" in fixed:
        forgets = forgets * (fixed["forgets"] < 0) + fixed["forgets"] * (fixed["forgets"] >= 0)
    As = np.empty((2, 2 * num_resources))
    interleave(As[0], 1 - learns, forgets.copy())
    interleave(As[1], learns.copy(), 1 - forgets)

    if "guesses" in fixed:
        guesses = fixed["guesses"] * (fixed["guesses"] < 0) + fixed["guesses"] * (fixed["guesses"] >= 0)
    if "slips" in fixed:
        slips = fixed["slips"] * (fixed["slips"] < 0) + fixed["slips"] * (fixed["slips"] >= 0)
    Bn = np.empty((2, 2 * num_subparts))
    interleave(Bn[0], 1 - guesses, guesses.copy())
    interleave(Bn[1], slips.copy(), 1 - slips)

    all_trans_softcounts = np.zeros((2, 2 * num_resources))
    all_emission_softcounts = np.zeros((2, 2 * num_subparts))
    all_initial_softcounts = np.zeros((2, 1))

    alpha_out = np.zeros((2, bigT))

    total_loglike = 0.0

    payload = {
        "As": As,
        "Bn": Bn,
        "initial_distn": initial_distn,
        "allresources": allresources,
        "starts": starts,
        "lengths": lengths,
        "num_resources": num_resources,
        "num_subparts": num_subparts,
        "alldata": alldata,
        "normalizeLengths": normalizeLengths,
        "alpha_out": alpha_out,
    }

    num_threads = cpu_count() if parallel else 1
    thread_counts = []
    for thread_num in range(num_threads):
        blocklen = 1 + ((num_sequences - 1) // num_threads)
        sequence_idx_start = int(blocklen * thread_num)
        sequence_idx_end = min(sequence_idx_start + blocklen, num_sequences)
        thread_input = {
            "sequence_idx_start": sequence_idx_start,
            "sequence_idx_end": sequence_idx_end,
            **payload,
        }
        thread_counts.append(thread_input)

    if parallel and num_threads > 1:
        with Pool(len(thread_counts)) as pool:
            results = pool.map(inner, thread_counts)
    else:
        results = [inner(item) for item in thread_counts]

    for trans, emission, init, loglike, alphas in results:
        total_loglike += float(loglike)
        all_trans_softcounts += trans
        all_emission_softcounts += emission
        all_initial_softcounts += init
        for sequence_start, length, alpha in alphas:
            alpha_out[:, sequence_start : sequence_start + length] += alpha

    all_trans_softcounts = all_trans_softcounts.flatten(order="F")
    all_emission_softcounts = all_emission_softcounts.flatten(order="F")
    return {
        "total_loglike": total_loglike,
        "all_trans_softcounts": np.reshape(all_trans_softcounts, (num_resources, 2, 2), order="C"),
        "all_emission_softcounts": np.reshape(all_emission_softcounts, (num_subparts, 2, 2), order="C"),
        "all_initial_softcounts": all_initial_softcounts,
        "alpha_out": alpha_out.flatten(order="F").reshape(alpha_out.shape, order="C"),
    }


def interleave(m, v1, v2):
    m[0::2], m[1::2] = v1, v2


def inner(x):
    (
        As,
        Bn,
        initial_distn,
        allresources,
        starts,
        lengths,
        num_resources,
        num_subparts,
        alldata,
        normalizeLengths,
        alpha_out,
        sequence_idx_start,
        sequence_idx_end,
    ) = (
        x["As"],
        x["Bn"],
        x["initial_distn"],
        x["allresources"],
        x["starts"],
        x["lengths"],
        x["num_resources"],
        x["num_subparts"],
        x["alldata"],
        x["normalizeLengths"],
        x["alpha_out"],
        x["sequence_idx_start"],
        x["sequence_idx_end"],
    )
    n_r, n_s = 2 * num_resources, 2 * num_subparts
    trans_softcounts_temp = np.zeros((2, n_r))
    emission_softcounts_temp = np.zeros((2, n_s))
    init_softcounts_temp = np.zeros((2, 1))
    loglike = 0.0
    alphas = []

    dot, sum_, log = np.dot, np.sum, np.log

    for sequence_index in range(sequence_idx_start, sequence_idx_end):
        sequence_start = starts[sequence_index] - 1
        t_len = lengths[sequence_index]

        likelihoods = np.ones((2, t_len))
        alpha = np.empty((2, t_len))
        for t in range(min(2, t_len)):
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    sl = Bn[:, 2 * n + int(data_temp == 2)]
                    likelihoods[:, t] *= np.where(sl == 0, 1, sl)

        alpha[:, 0] = _normalize_two_state(initial_distn * likelihoods[:, 0])
        norm = float(sum_(initial_distn * likelihoods[:, 0]))
        loglike += log(max(norm, _NUMERIC_EPS)) / (t_len if normalizeLengths else 1)

        if t_len >= 2:
            resources_temp = allresources[sequence_start]
            k = 2 * (resources_temp - 1)
            alpha[:, 1] = dot(As[0:2, k : k + 2], alpha[:, 0]) * likelihoods[:, 1]
            alpha[:, 1] = _normalize_two_state(alpha[:, 1])
            norm = float(sum_(dot(As[0:2, k : k + 2], alpha[:, 0]) * likelihoods[:, 1]))
            loglike += log(max(norm, _NUMERIC_EPS)) / (t_len if normalizeLengths else 1)

        for t in range(2, t_len):
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    sl = Bn[:, 2 * n + int(data_temp == 2)]
                    likelihoods[:, t] *= np.where(sl == 0, 1, sl)
            resources_temp = allresources[sequence_start + t - 1]
            k = 2 * (resources_temp - 1)
            alpha[:, t] = dot(As[0:2, k : k + 2], alpha[:, t - 1]) * likelihoods[:, t]
            alpha[:, t] = _normalize_two_state(alpha[:, t])
            norm = float(sum_(dot(As[0:2, k : k + 2], alpha[:, t - 1]) * likelihoods[:, t]))
            loglike += log(max(norm, _NUMERIC_EPS)) / (t_len if normalizeLengths else 1)

        gamma = np.empty((2, t_len))
        gamma[:, t_len - 1] = alpha[:, t_len - 1].copy()
        as_temp = As.copy()
        first = True
        for t in range(t_len - 2, -1, -1):
            resources_temp = allresources[sequence_start + t]
            k = 2 * (resources_temp - 1)
            a_mat = as_temp[0:2, k : k + 2]
            pair = a_mat.copy()
            pair[0] *= alpha[:, t]
            pair[1] *= alpha[:, t]
            dotted = dot(a_mat, alpha[:, t])
            gamma_t = gamma[:, t + 1]
            pair[:, 0] = _safe_divide(pair[:, 0] * gamma_t, dotted)
            pair[:, 1] = _safe_divide(pair[:, 1] * gamma_t, dotted)
            trans_softcounts_temp[0:2, k : k + 2] += pair
            gamma[:, t] = sum_(pair, axis=0)
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    emission_softcounts_temp[:, 2 * n + int(data_temp == 2)] += gamma[:, t]
                if first:
                    data_temp_p = alldata[n][sequence_start + (t_len - 1)]
                    if data_temp_p:
                        emission_softcounts_temp[:, 2 * n + int(data_temp_p == 2)] += gamma[:, t_len - 1]
            first = False

        init_softcounts_temp += gamma[:, 0].reshape((2, 1))
        alphas.append((sequence_start, t_len, alpha))

    return [trans_softcounts_temp, emission_softcounts_temp, init_softcounts_temp, loglike, alphas]
