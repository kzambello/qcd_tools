# Copyright 2025 Kevin Zambello
#
# This program is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Module providing utilities for the multi-point Pade' approximation."""

import io
import os
import time
from multiprocessing import Pool
import numpy as np
from scipy import optimize
from scipy import signal
import sympy as sp


def fit(
    mu_in,
    chis_in,
    dchis_in,
    constraints,
    ncoeffs_num,
    ncoeffs_den,
    nsamples=1,
    seed=0,
    nonlinear_fit=True,
    verbosity=1,
):
    """
    Find a multi-point Pade' approximation R^N_M(mu) for the real function of real variable chi(mu).

    Args:
            mu_in (float numpy array):            numpy array of mu values

            chis_in (list of float numpy arrays): the n-th element of the list, chis_in[n], is a numpy array
                                                  containing the n-th order derivative of the function
                                                  chi evaluted at mu_in

            constraints (int numpy array):         numpy array defining the constraints for solving the multi-point Pade';
                                                   the rational approximant is required to be equal to the function
                                                   chi at mu_in[n] up to order contraints[n]

            ncoeffs_num (int):                     number of coefficients in the numerator

            ncoeffs_den (int):                     number of coefficients in the denominator

            nsamples (int):                        number of sample for bootstrap

            seed (int):                            seed for bootstrap

            nonlinear_fit (boolean):               if True solve the nonlinear multi-point Pade' problem
                                                   using the generalized chi2 approach, if False solve the
                                                   linearized problem

            verbosity (int):                       verbosity level (0,1,2,3)

    Returns:

            [sol_f0_lst, roots_num_arr_lst, roots_den_arr_lst, chi_square_lst, residues_arr_lst]

               WHERE

               sol_f0_lst (list of lambda functions):        Pade' approximant for chi

               roots_num_arr_lst (list of comples numpy arrays): zeros of the numerator (remapped to real mu)

               roots_den_arr_lst (list of comples numpy arrays)  zeros of the denominator (remapped to real mu)

               chi_square_lst (list of floats)               chi2/ndof

               residues_arr_lst (list of comples numpy arrays)   complex residues

    """

    output_buffer = io.StringIO()

    def print_to_buffer(string):
        print(string, file=output_buffer)

    if seed == 0:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    chis_resampled = []

    norders = len(chis_in)

    for i in np.arange(nsamples):
        if i == 0:
            tmp = []
            for j in np.arange(norders):
                tmp = tmp + [chis_in[j]]
            chis_resampled.append(tmp)
        else:
            tmp = []
            for j in np.arange(norders):
                tmp = tmp + [chis_in[j] + dchis_in[j] * np.random.randn(len(mu_in))]
            chis_resampled.append(tmp)

    npoints = len(mu_in)

    nconstraints = (
        constraints.sum() + 1
    )  # add 1 since we always have one implicit constraint,
    # i.e. we require that the first coefficient at the denominator is 1

    x = sp.symbols("x")
    C = sp.symarray("C", ncoeffs_num + ncoeffs_den)
    A = C[0:ncoeffs_num]
    B = C[ncoeffs_num : ncoeffs_num + ncoeffs_den]
    p = sp.Poly(A, x)
    q = sp.Poly(B, x)
    pq = p / q

    if verbosity > 0:
        print_to_buffer(f">>> Proc {os.getpid()} <<<")
        # print_to_buffer("--------------------------------------------------")
        print_to_buffer(
            f"Number of points = {npoints}\nNumber of constraints = {nconstraints} (for {len(constraints)} points)\nNumber of coefficients = {(ncoeffs_num + ncoeffs_den)}\nNumber of degrees of freedom = {(nconstraints - ncoeffs_num - ncoeffs_den)}"
        )
        print_to_buffer(f"\nRational function: {pq}")

        if (ncoeffs_num + ncoeffs_den) > nconstraints:
            print_to_buffer(
                "\nWARNING: the number of coefficients is greater than the number of constraints."
            )
        print_to_buffer("--------------------------------------------------")
        print(output_buffer.getvalue(), flush=True)
        output_buffer.seek(0)
        output_buffer.truncate(0)

    chi2_ndof = nconstraints - ncoeffs_num - ncoeffs_den
    if chi2_ndof <= 0:
        chi2_ndof = 1

    # pylint: disable=not-callable
    f = sp.Function("f")(x)
    myrhs = p.as_expr()
    mylhs = f * (q.as_expr())

    sol_f0_lst = []
    # sol_f1_arr_lst = []
    roots_num_arr_lst = []
    roots_den_arr_lst = []
    chi_square_lst = []
    residues_arr_lst = []

    for current_sample in np.arange(nsamples):
        if verbosity > 0:
            print_to_buffer(
                f">>> Proc {os.getpid()} [sample #{current_sample+1} of {nsamples}] <<<"
            )
            # print_to_buffer("--------------------------------------------------")

        chis = chis_resampled[current_sample]
        dchis = dchis_in

        eqs = [C[ncoeffs_num + ncoeffs_den - 1] - 1.0]
        mychi2 = 0.0
        for i in np.arange(npoints):
            for j in np.arange(norders):
                if constraints[i] > j:
                    dmyrhs = myrhs.diff((x, j))
                    dmylhs = mylhs.diff((x, j))
                    myeq = (dmyrhs - dmylhs).subs(x, mu_in[i])
                    for k in np.arange(norders):
                        myexpr = (f.diff((x, k))).subs(x, mu_in[i])
                        myeq = myeq.subs(myexpr, chis[k][i])
                    eqs.append(myeq)

                    mychi2 = (
                        mychi2
                        + (
                            ((sp.diff(pq, x, j)).subs(x, mu_in[i]) - chis[j][i])
                            / dchis[j][i]
                        )
                        ** 2
                    )

        (A, b) = sp.linear_eq_to_matrix(eqs, C)
        A = np.array(A).astype(np.float64)  # float128 are not supported by linalg
        b = np.array(b).astype(np.float64)

        mychi2fun = sp.lambdify(C, mychi2)

        def mychi2fun_v(z):
            return mychi2fun(*z) / chi2_ndof

        if nconstraints == (ncoeffs_num + ncoeffs_den):
            if verbosity > 1:
                print_to_buffer(f"... condition number = {np.linalg.cond(A)} ...")
                print_to_buffer("... solving linearized problem (using solve) ...")
            sol = np.linalg.solve(A, b)
        else:
            if verbosity > 1:
                print_to_buffer(f"... condition number = {np.linalg.cond(A)} ...")
                print_to_buffer("... solving linearized problem (using lstsq) ...")
            sol, residuals, rank, singval = np.linalg.lstsq(A, b, rcond=None)

        sol = np.squeeze(sol)
        if verbosity > 1:
            print_to_buffer(
                f"... done: res = {np.linalg.norm(np.dot(A, sol.reshape(len(sol), 1)) - b)}, chi2 = {mychi2fun_v(sol)} ..."
            )

        ### minimize generalized chi2
        if nonlinear_fit is True:
            if verbosity > 1:
                print_to_buffer(
                    f"... minimizing generalized chi2 = {mychi2fun_v(sol)} ..."
                )
            C0 = sol
            mymethod = "Nelder-Mead"
            myoptions = {
                "disp": False,
                "maxiter": 10000,
                "maxfev": 10000,
                "adaptive": True,
            }
            for i in np.arange(5):
                r = optimize.minimize(
                    mychi2fun_v, C0, method=mymethod, options=myoptions
                )
                C0 = r.x
            sol = C0
            if verbosity > 1:
                print_to_buffer(
                    f"... done: res = {np.linalg.norm(np.dot(A, sol.reshape(len(sol), 1)) - b)}, chi2 = {mychi2fun_v(sol)} ...\n"
                )
        ###

        mypq = pq
        for i in np.arange(ncoeffs_num + ncoeffs_den):
            mypq = mypq.subs(C[i], sol[i])

        if verbosity > 0:
            print_to_buffer(f"Rational function: {mypq}")
            print_to_buffer("--------------------------------------------------")

        if verbosity > 2:
            print_to_buffer(
                f"\n>>> Proc {os.getpid()} [sample #{current_sample+1} of {nsamples}] <<<"
            )
            # print_to_buffer("--------------------------------------------------")
            for j in np.arange(norders):
                print_to_buffer(f"=== cross-check for order {j} ===")
                myf = sp.lambdify(x, sp.diff(mypq, (x, j)))
                for i in np.arange(npoints):
                    print_to_buffer(
                        "[mu = %.1f] %+.3e vs %+.3e (err: %.3e err/f: %.3e err/df: %.3e)"
                        % (
                            mu_in[i],
                            chis[j][i],
                            myf(mu_in[i]),
                            np.abs(myf(mu_in[i]) - chis[j][i]),
                            np.abs(myf(mu_in[i]) - chis[j][i]) / np.abs(chis[j][i]),
                            np.abs(myf(mu_in[i]) - chis[j][i]) / np.abs(dchis[j][i]),
                        )
                    )

        myp = p.all_coeffs()
        myq = q.all_coeffs()
        for j in np.arange(len(myp)):
            for i in np.arange(ncoeffs_num + ncoeffs_den):
                myp[j] = myp[j].subs(C[i], sol[i])
        for j in np.arange(len(myq)):
            for i in np.arange(ncoeffs_num + ncoeffs_den):
                myq[j] = myq[j].subs(C[i], sol[i])
        for j in np.arange(ncoeffs_num):
            myp[j] = np.float64(myp[j])
        for j in np.arange(ncoeffs_den):
            myq[j] = np.float64(myq[j])

        sol_f0 = sp.lambdify(x, mypq)
        # sol_f1 = sp.lambdify(x, sp.diff(mypq, (x, 1)))

        sol_f0_lst.append(sol_f0)
        # sol_f1_arr_lst.append(sol_f1)

        roots_num = np.roots(myp).astype(complex)
        roots_den = np.roots(myq).astype(complex)

        roots_num_arr_lst.append(roots_num)
        roots_den_arr_lst.append(roots_den)

        chi_square = mychi2fun_v(sol)

        chi_square_lst.append(chi_square)

        myfact = 1.0
        for j in np.arange(len(myp)):
            myp[len(myp) - 1 - j] = 1.0j * myp[len(myp) - 1 - j] * myfact
            myfact = myfact * (-1.0j)
        myfact = 1.0
        for j in np.arange(len(myq)):
            myq[len(myq) - 1 - j] = myq[len(myq) - 1 - j] * myfact
            myfact = myfact * (-1.0j)
        (residues, _, _) = signal.residue(myp, myq)

        residues_arr_lst.append(residues)

        if verbosity > 2:
            print_to_buffer("=== zeros (remapped to real mu) ===")
            print_to_buffer(1.0j * roots_num)
            print_to_buffer("=== poles (remapped to real mu) ===")
            print_to_buffer(1.0j * roots_den)
            print_to_buffer("=== residues ===")
            print_to_buffer(residues)
            print_to_buffer("--------------------------------------------------")

        if verbosity > 0:
            print(output_buffer.getvalue(), flush=True)
            output_buffer.seek(0)
            output_buffer.truncate(0)

    return [
        sol_f0_lst,
        [1.0j * x for x in roots_num_arr_lst],
        [1.0j * x for x in roots_den_arr_lst],
        chi_square_lst,
        residues_arr_lst,
    ]


def mpfit(
    mu_in,
    chis_in,
    dchis_in,
    constraints,
    ncoeffs_num,
    ncoeffs_den,
    nsamples=1,
    seed=0,
    nonlinear_fit=True,
    verbosity=1,
    ntasks=2,
):
    """
    Parallel fit. Be aware that since Pool.map() does not support lambda functions, it returns None as sol_f0_lst.
    """

    global my_task

    def my_task(my_task_args):
        [_, roots_num_arr_lst, roots_den_arr_lst, chi_square_lst, residues_arr_lst] = (
            fit(
                mu_in,
                chis_in,
                dchis_in,
                constraints,
                ncoeffs_num,
                ncoeffs_den,
                nsamples=my_task_args[0],
                seed=my_task_args[1],
                nonlinear_fit=nonlinear_fit,
                verbosity=verbosity,
            )
        )

        return [roots_num_arr_lst, roots_den_arr_lst, chi_square_lst, residues_arr_lst]

    ntasks = min(ntasks, nsamples)

    if seed == 0:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    if ntasks == 1:
        nsamples_splitted = np.ones(1, dtype=int) * nsamples
    else:
        nsamples_splitted = np.ones(ntasks - 1, dtype=int) * int(
            np.floor(nsamples / ntasks)
        )
        nsamples_splitted = np.append(
            nsamples_splitted, nsamples - np.sum(nsamples_splitted)
        )

    my_task_args = []
    for task_id in np.arange(ntasks):
        my_task_args = my_task_args + [
            (nsamples_splitted[task_id], np.random.randint(1, 1000000))
        ]

    with Pool(ntasks) as p:
        res = p.map(my_task, my_task_args)

    roots_num_arr_lst = []
    roots_den_arr_lst = []
    chi_square_lst = []
    residues_arr_lst = []

    for task_id in np.arange(ntasks):
        roots_num_arr_lst = roots_num_arr_lst + res[task_id][0]
        roots_den_arr_lst = roots_den_arr_lst + res[task_id][1]
        chi_square_lst = chi_square_lst + res[task_id][2]
        residues_arr_lst = residues_arr_lst + res[task_id][3]

    return [
        None,
        roots_num_arr_lst,
        roots_den_arr_lst,
        chi_square_lst,
        residues_arr_lst,
    ]


def simplify_roots(roots_num_arr, roots_den_arr, residues_arr, tol_cancellation):
    """
    Remove zero/pole pairs.
    """

    roots_num_arr_simplified = np.copy(roots_num_arr)
    roots_den_arr_simplified = np.copy(roots_den_arr)
    residues_arr_simplified = np.copy(residues_arr)

    for _ in np.arange(roots_den_arr_simplified.size):
        zdeleted = 0
        pdeleted = 0
        for idxp in np.arange(roots_den_arr_simplified.size):
            for idxz in np.arange(roots_num_arr_simplified.size):
                if idxz < (roots_num_arr_simplified.size - zdeleted) and idxp < (
                    roots_den_arr_simplified.size - pdeleted
                ):
                    if (
                        np.abs(
                            roots_den_arr_simplified[idxp]
                            - roots_num_arr_simplified[idxz]
                        )
                        < tol_cancellation
                    ):
                        roots_num_arr_simplified = np.delete(
                            roots_num_arr_simplified, idxz
                        )
                        roots_den_arr_simplified = np.delete(
                            roots_den_arr_simplified, idxp
                        )
                        residues_arr_simplified = np.delete(
                            residues_arr_simplified, idxp
                        )
                        zdeleted = zdeleted + 1
                        pdeleted = pdeleted + 1

    return [roots_num_arr_simplified, roots_den_arr_simplified, residues_arr_simplified]


def squeeze_roots_lst(
    roots_num_arr_lst,
    roots_den_arr_lst,
    chi_square_lst,
    residues_arr_lst,
    chi_threshold=None,
    tol_cancellation=None,
    keep_only_one=None,
):
    """
    Merge zeros/poles from different samples optionally discarding zeros/poles from bad fits, cancelling zero/pole pairs, keeping only one pole.
    """

    roots_num_arr_lst_squeezed = np.array([])
    roots_den_arr_lst_squeezed = np.array([])
    residues_arr_lst_squeezed = np.array([])

    for n in np.arange(len(roots_num_arr_lst)):
        if chi_threshold is None or chi_square_lst[n] < chi_threshold:
            if tol_cancellation is None:
                tmp_a = roots_num_arr_lst[n]
                tmp_b = roots_den_arr_lst[n]
                tmp_c = residues_arr_lst[n]
            else:
                [tmp_a, tmp_b, tmp_c] = simplify_roots(
                    roots_num_arr_lst[n],
                    roots_den_arr_lst[n],
                    residues_arr_lst[n],
                    tol_cancellation,
                )

            if keep_only_one is not None:
                idx = np.squeeze(np.argwhere((tmp_b.real > 0) & (tmp_b.imag > 0)))
                tmp_b = np.array(tmp_b[idx])
                tmp_c = np.array(tmp_c[idx])
                if np.size(tmp_b) == 0:
                    tmp_b = np.array([])
                    tmp_c = np.array([])
                else:
                    if np.size(tmp_b) == 1:
                        tmp_b = np.array([tmp_b])
                        tmp_c = np.array([tmp_c])
                    idx = np.argmin(np.abs(tmp_b - keep_only_one))
                    tmp_b = np.array(tmp_b[idx])
                    if np.size(tmp_b) == 0:
                        tmp_b = np.array([])
                        tmp_c = np.array([])

            roots_num_arr_lst_squeezed = np.append(roots_num_arr_lst_squeezed, tmp_a)
            roots_den_arr_lst_squeezed = np.append(roots_den_arr_lst_squeezed, tmp_b)
            residues_arr_lst_squeezed = np.append(residues_arr_lst_squeezed, tmp_c)

    return [
        roots_num_arr_lst_squeezed,
        roots_den_arr_lst_squeezed,
        residues_arr_lst_squeezed,
    ]
