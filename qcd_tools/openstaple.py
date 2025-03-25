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

"""Module providing utilities for OpenStaPLE."""

import numpy as np
import mpmath
from mpmath import mp

mp.dps = 40


def do_jk(src, nblocks, my_function=np.mean):
    """Block jackknife."""

    data = np.copy(src)

    blocksize = int(data.size / nblocks)

    imax = int(np.floor(np.size(data) / blocksize) * blocksize)
    b = np.reshape(data[0:imax], (-1, blocksize))

    jksamples = imax / blocksize
    jkestimates = []

    for k in np.arange(imax / blocksize):
        jksample = np.reshape(np.delete(b, int(k), axis=0), (1, -1))[0]
        jkestimates = jkestimates + [my_function(jksample)]
    jkestimates = np.array(jkestimates)

    jkmean = np.mean(jkestimates)
    jkerr = np.sqrt(
        ((jksamples - 1.0) / jksamples) * np.sum((jkestimates - my_function(data)) ** 2)
    )

    return jkmean, jkerr


def jk_mean(src, nblocks, my_function=np.mean):
    """Wrapper for block jackknife."""
    return do_jk(src, nblocks, my_function=my_function)[0]


def jk_err(src, nblocks, my_function=np.mean):
    """Wrapper for block jackknife."""
    return do_jk(src, nblocks, my_function=my_function)[1]


def do_mhist(
    MyNs,
    MyNt,
    beta,
    plaq,
    rect,
    nbeta_mhist,
    each_mhist,
    min_iter_mhist,
    max_iter_mhist,
    tol_mhist,
):
    """Multiple histogram reweighting (for TLSM gauge action)."""

    nbeta = beta.size

    def S(bbeta, pplaq, rrect, nns, nnt):
        return -(
            5.0 / 6.0 * pplaq * nns**3 * nnt * 4 * 3
            - 1.0 / 12.0 * rrect * nns**3 * nnt * 4 * 3
        )

    logZ = np.array([1.0e3 for x in beta], dtype=np.float128)

    def calc_logZ(mybeta):
        res = 0.0
        for i in np.arange(nbeta):
            for a in np.arange(plaq[i][0::each_mhist].size):
                myS = S(
                    beta[i],
                    plaq[i][0::each_mhist][a],
                    rect[i][0::each_mhist][a],
                    MyNs,
                    MyNt,
                )
                den = 0.0
                for j in np.arange(nbeta):
                    den = den + plaq[j][0::each_mhist].size * mp.exp(
                        (-beta[j] + mybeta) * myS - logZ[j]
                    )
                res = res + 1.0 / den
        return mp.log(res)

    def calc_plaq(mybeta):
        res = 0.0
        for i in np.arange(nbeta):
            for a in np.arange(plaq[i][0::each_mhist].size):
                myS = S(
                    beta[i],
                    plaq[i][0::each_mhist][a],
                    rect[i][0::each_mhist][a],
                    MyNs,
                    MyNt,
                )
                den = 0.0
                for j in np.arange(nbeta):
                    den = den + plaq[j][0::each_mhist].size * mp.exp(
                        (-beta[j] + mybeta) * myS - logZ[j]
                    ) / (plaq[i][0::each_mhist][a])
                res = res + 1.0 / den
        res = res / mp.exp(calc_logZ(mybeta))
        return res

    def calc_plaq2(mybeta):
        res = 0.0
        for i in np.arange(nbeta):
            for a in np.arange(plaq[i][0::each_mhist].size):
                myS = S(
                    beta[i],
                    plaq[i][0::each_mhist][a],
                    rect[i][0::each_mhist][a],
                    MyNs,
                    MyNt,
                )
                den = 0.0
                for j in np.arange(nbeta):
                    den = den + plaq[j][0::each_mhist].size * mp.exp(
                        (-beta[j] + mybeta) * myS - logZ[j]
                    ) / (plaq[i][0::each_mhist][a] ** 2)
                res = res + 1.0 / den
        res = res / mp.exp(calc_logZ(mybeta))
        return res

    def calc_plaq4(mybeta):
        res = 0.0
        for i in np.arange(nbeta):
            for a in np.arange(plaq[i][0::each_mhist].size):
                myS = S(
                    beta[i],
                    plaq[i][0::each_mhist][a],
                    rect[i][0::each_mhist][a],
                    MyNs,
                    MyNt,
                )
                den = 0.0
                for j in np.arange(nbeta):
                    den = den + plaq[j][0::each_mhist].size * mp.exp(
                        (-beta[j] + mybeta) * myS - logZ[j]
                    ) / (plaq[i][0::each_mhist][a] ** 4)
                res = res + 1.0 / den
        res = res / mp.exp(calc_logZ(mybeta))
        return res

    for n in np.arange(max_iter_mhist):
        # calc Z_j
        tmp = np.array([1.0 for x in beta], dtype=np.float128)
        deltaZ = 0.0
        for i in np.arange(nbeta):
            tmp[i] = calc_logZ(beta[i])
            deltaZ = (
                deltaZ + ((logZ[i] - tmp[i]) / np.max([np.abs(logZ[i]), 1.0e-16])) ** 2
            )
            # print(f"beta = {beta[i]}, logZold = {logZ[i]}, logZnew = {tmp[i]}")
        deltaZ = np.sqrt(deltaZ) / nbeta
        # update Z_j
        for i in np.arange(nbeta):
            logZ[i] = tmp[i]
        print(f"iteration {n}   deltaZ = {deltaZ}")
        if deltaZ < tol_mhist and n > min_iter_mhist:
            break

    beta_mhist = np.linspace(np.min(beta), np.max(beta), nbeta_mhist)
    plaq_mhist = np.zeros(nbeta_mhist)
    plaq4_mhist = np.zeros(nbeta_mhist)
    suscplaq_mhist = np.zeros(nbeta_mhist)
    bindplaq_mhist = np.zeros(nbeta_mhist)

    for n in np.arange(nbeta_mhist):
        plaq_mhist[n] = calc_plaq(beta_mhist[n])
        plaq2_mhist[n] = calc_plaq2(beta_mhist[n])
        plaq4_mhist[n] = plaq4_mybeta(beta_mhist[n])
        suscplaq_mhist[n] = (MyNs**3) * MyNt * (plaq2_mhist[n] - plaq_mhist[n] ** 2)
        bindplaq_mhist[n] = plaq4_mhist[n] / (plaq2_mhist[n] ** 2)

    return (
        beta_mhist,
        plaq_mhist,
        plaq2_mhist,
        plaq4_mhist,
        suscplaq_mhist,
        bindplaq_mhist,
    )


def do_mhist_jack(
    MyNs,
    MyNt,
    beta,
    plaq,
    rect,
    nbeta_mhist,
    each_mhist,
    min_iter_mhist,
    max_iter_mhist,
    tol_mhist,
    nblocks,
):
    """Multiple histogram reweighting (for TLSM gauge action) with block jackknife."""

    nbeta = beta.size

    def S(bbeta, pplaq, rrect, nns, nnt):
        return -(
            5.0 / 6.0 * pplaq * nns**3 * nnt * 4 * 3
            - 1.0 / 12.0 * rrect * nns**3 * nnt * 4 * 3
        )

    logZ = np.array([1.0e3 for x in beta], dtype=np.float128)

    blocksizes = np.array(np.array([el.size for el in plaq]) / nblocks, dtype=int)

    beta_mhist = np.linspace(np.min(beta), np.max(beta), nbeta_mhist)
    plaq_mhist_jack = np.zeros((nblocks, nbeta_mhist))
    plaq2_mhist_jack = np.zeros((nblocks, nbeta_mhist))
    plaq4_mhist_jack = np.zeros((nblocks, nbeta_mhist))
    suscplaq_mhist_jack = np.zeros((nblocks, nbeta_mhist))
    bindplaq_mhist_jack = np.zeros((nblocks, nbeta_mhist))

    for blk in np.arange(nblocks):
        print(f"=== BLOCK {blk}/{nblocks} ===")
        # keep previously calculated logZ as the initial estimate

        plaq_jack = []
        rect_jack = []

        for j in np.arange(nbeta):
            plaq_jack_copy = np.copy(plaq[j])
            rect_jack_copy = np.copy(rect[j])

            blocksize = blocksizes[j]
            imax = int(np.floor(np.size(plaq_jack_copy) / blocksize) * blocksize)
            bplaq = np.reshape(plaq_jack_copy[0:imax], (-1, blocksize))
            imax = int(np.floor(np.size(rect_jack_copy) / blocksize) * blocksize)
            brect = np.reshape(rect_jack_copy[0:imax], (-1, blocksize))
            plaq_jack = plaq_jack + [
                np.reshape(np.delete(bplaq, int(blk), axis=0), (1, -1))[0]
            ]
            rect_jack = rect_jack + [
                np.reshape(np.delete(brect, int(blk), axis=0), (1, -1))[0]
            ]

        def calc_logZ(mybeta):
            res = 0.0
            for i in np.arange(nbeta):
                for a in np.arange(plaq_jack[i][0::each_mhist].size):
                    myS = S(
                        beta[i],
                        plaq_jack[i][0::each_mhist][a],
                        rect_jack[i][0::each_mhist][a],
                        MyNs,
                        MyNt,
                    )
                    den = 0.0
                    for j in np.arange(nbeta):
                        den = den + plaq_jack[j][0::each_mhist].size * mp.exp(
                            (-beta[j] + mybeta) * myS - logZ[j]
                        )
                    res = res + 1.0 / den
            return mp.log(res)

        def calc_plaq(mybeta):
            res = 0.0
            for i in np.arange(nbeta):
                for a in np.arange(plaq_jack[i][0::each_mhist].size):
                    myS = S(
                        beta[i],
                        plaq_jack[i][0::each_mhist][a],
                        rect_jack[i][0::each_mhist][a],
                        MyNs,
                        MyNt,
                    )
                    den = 0.0
                    for j in np.arange(nbeta):
                        den = den + plaq_jack[j][0::each_mhist].size * mp.exp(
                            (-beta[j] + mybeta) * myS - logZ[j]
                        ) / (plaq_jack[i][0::each_mhist][a])
                    res = res + 1.0 / den
            res = res / mp.exp(calc_logZ(mybeta))
            return res

        def calc_plaq2(mybeta):
            res = 0.0
            for i in np.arange(nbeta):
                for a in np.arange(plaq_jack[i][0::each_mhist].size):
                    myS = S(
                        beta[i],
                        plaq_jack[i][0::each_mhist][a],
                        rect_jack[i][0::each_mhist][a],
                        MyNs,
                        MyNt,
                    )
                    den = 0.0
                    for j in np.arange(nbeta):
                        den = den + plaq_jack[j][0::each_mhist].size * mp.exp(
                            (-beta[j] + mybeta) * myS - logZ[j]
                        ) / (plaq_jack[i][0::each_mhist][a] ** 2)
                    res = res + 1.0 / den
            res = res / mp.exp(calc_logZ(mybeta))
            return res

        def calc_plaq4(mybeta):
            res = 0.0
            for i in np.arange(nbeta):
                for a in np.arange(plaq_jack[i][0::each_mhist].size):
                    myS = S(
                        beta[i],
                        plaq_jack[i][0::each_mhist][a],
                        rect_jack[i][0::each_mhist][a],
                        MyNs,
                        MyNt,
                    )
                    den = 0.0
                    for j in np.arange(nbeta):
                        den = den + plaq_jack[j][0::each_mhist].size * mp.exp(
                            (-beta[j] + mybeta) * myS - logZ[j]
                        ) / (plaq_jack[i][0::each_mhist][a] ** 4)
                    res = res + 1.0 / den
            res = res / mp.exp(calc_logZ(mybeta))
            return res

        for n in np.arange(max_iter_mhist):
            # calc Z_j
            tmp = np.array([1.0 for x in beta], dtype=np.float128)
            deltaZ = 0.0
            for i in np.arange(nbeta):
                tmp[i] = calc_logZ(beta[i])
                deltaZ = (
                    deltaZ
                    + ((logZ[i] - tmp[i]) / np.max([np.abs(logZ[i]), 1.0e-16])) ** 2
                )
                # print(f"beta = {beta[i]}, logZold = {logZ[i]}, logZnew = {tmp[i]}")
            deltaZ = np.sqrt(deltaZ) / nbeta
            # update Z_j
            for i in np.arange(nbeta):
                logZ[i] = tmp[i]
            print(f"iteration {n}   deltaZ = {deltaZ}")
            if deltaZ < tol_mhist and n > min_iter_mhist:
                break

        for n in np.arange(nbeta_mhist):
            plaq_mhist_jack[blk][n] = calc_plaq(beta_mhist[n])
            plaq2_mhist_jack[blk][n] = calc_plaq2(beta_mhist[n])
            plaq4_mhist_jack[blk][n] = calc_plaq4(beta_mhist[n])
            suscplaq_mhist_jack[blk][n] = (
                (MyNs**3)
                * MyNt
                * (plaq2_mhist_jack[blk][n] - plaq_mhist_jack[blk][n] ** 2)
            )
            bindplaq_mhist_jack[blk][n] = plaq4_mhist_jack[blk][n] / (
                plaq2_mhist_jack[blk][n] ** 2
            )

    plaq_mhist = np.zeros(nbeta_mhist)
    plaq2_mhist = np.zeros(nbeta_mhist)
    plaq4_mhist = np.zeros(nbeta_mhist)
    suscplaq_mhist = np.zeros(nbeta_mhist)
    bindplaq_mhist = np.zeros(nbeta_mhist)

    dplaq_mhist = np.zeros(nbeta_mhist)
    dplaq2_mhist = np.zeros(nbeta_mhist)
    dplaq4_mhist = np.zeros(nbeta_mhist)
    dsuscplaq_mhist = np.zeros(nbeta_mhist)
    dbindplaq_mhist = np.zeros(nbeta_mhist)

    for n in np.arange(nbeta_mhist):
        p = plaq_mhist_jack[:, n]
        p2 = plaq2_mhist_jack[:, n]
        p4 = plaq4_mhist_jack[:, n]
        sp = suscplaq_mhist_jack[:, n]
        bp = bindplaq_mhist_jack[:, n]

        plaq_mhist[n] = np.mean(p)
        plaq2_mhist[n] = np.mean(p2)
        plaq4_mhist[n] = np.mean(p4)
        suscplaq_mhist[n] = np.mean(sp)
        bindplaq_mhist[n] = np.mean(bp)

        dplaq_mhist[n] = np.std(p) * np.sqrt(nblocks-1)
        dplaq2_mhist[n] = np.std(p2) * np.sqrt(nblocks-1)
        dplaq4_mhist[n] = np.std(p4) * np.sqrt(nblocks-1)
        dsuscplaq_mhist[n] = np.std(sp) * np.sqrt(nblocks-1)
        dbindplaq_mhist[n] = np.std(bp) * np.sqrt(nblocks-1)

    return (
        beta_mhist,
        plaq_mhist,
        dplaq_mhist,
        plaq2_mhist,
        dplaq2_mhist,
        plaq4_mhist,
        dplaq4_mhist,
        suscplaq_mhist,
        dsuscplaq_mhist,
        bindplaq_mhist,
        dbindplaq_mhist,
        plaq_mhist_jack,
        plaq2_mhist_jack,
        plaq4_mhist_jack,
        suscplaq_mhist_jack,
        bindplaq_mhist_jack
    )
