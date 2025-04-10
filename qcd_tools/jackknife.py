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

"""Module providing block jackknife."""

import numpy as np


def jk(srcs, nblocks, my_function=np.mean):
    """Block jackknife."""

    data = []
    for n in np.arange(len(srcs)):
        src = np.copy(srcs[n])
        data.append(src)

    blocksize = int(data[0].size / nblocks)

    imax = int(np.floor(np.size(data[0]) / blocksize) * blocksize)

    num_jksamples = imax / blocksize
    jkestimates = []

    for k in np.arange(num_jksamples):
        jksample = []
        for n in np.arange(len(srcs)):
            blocked_data = np.reshape(data[n][0:imax], (-1, blocksize))
            jksample.append(
                np.reshape(np.delete(blocked_data, int(k), axis=0), (1, -1))[0]
            )
        jkestimates = jkestimates + [my_function(*jksample)]
    jkestimates = np.array(jkestimates)

    jkmean = np.mean(jkestimates)
    jkerr = np.sqrt(
        ((num_jksamples - 1.0) / num_jksamples)
        * np.sum((jkestimates - my_function(*data)) ** 2)
    )

    return jkmean, jkerr


def jk_mean(srcs, nblocks, my_function=np.mean):
    """Wrapper for block jackknife."""
    return jk(srcs, nblocks, my_function=my_function)[0]


def jk_err(srcs, nblocks, my_function=np.mean):
    """Wrapper for block jackknife."""
    return jk(srcs, nblocks, my_function=my_function)[1]
