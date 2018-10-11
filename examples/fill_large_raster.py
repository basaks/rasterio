from pathlib import Path
import numpy as np
import rasterio as rio
from rasterio.fill import fillnodata
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



input_raster = '/home/sudiptra/repos/uncover-ml/relief_apsect.tif'
src = Path(input_raster)
dest = src.with_suffix('.filled.tif')


if rank == 0:
    if dest.exists():
        raise FileExistsError('Output file {} exists'.format(dest.as_posix()))
    else:  # copy
        dest.write_bytes(src.read_bytes())

ids = rio.open(src, 'r')

data = ids.read(1, masked=True)

ods = rio.open(dest, 'r+')
ods.write_band(1, np.empty_like(data, dtype=np.float32))

num_rows, num_cols = 4, 4

rows = np.linspace(0, data.shape[0], num_rows, dtype=int)
cols = np.linspace(0, data.shape[1], num_cols, dtype=int)

kernel_size = 5

from collections import namedtuple

Tile = namedtuple('Tile', ['data', 'window'])

data = ids.read(1, masked=True)


def _fill_tile(r, c):
    r_buffer_b = kernel_size if r != 0 else 0
    r_buffer_t = kernel_size if r != num_rows - 2 else 0
    c_buffer_l = kernel_size if c != 0 else 0
    c_buffer_r = kernel_size if c != num_cols - 2 else 0
    window_write = ((rows[r], rows[r + 1]),
                    (cols[c], cols[c + 1]))
    window_read = ((rows[r] - r_buffer_b, rows[r + 1] + r_buffer_t),
                   (cols[c] - c_buffer_l, cols[c + 1] + c_buffer_r))
    print('Write window {}'.format(window_write))
    print('Read window due to patch {}'.format(window_read))
    tile = data[window_read[0][0]:window_read[0][1], window_read[1][0]:window_read[1][1]]

    # TODO: Handle easy all maksed condition

    data_masked = fillnodata(tile, mask=None, max_search_distance=kernel_size)
    print(r_buffer_b, r_buffer_t, c_buffer_l, c_buffer_r)

    # return r_buffer_b, r_buffer_t, c_buffer_l, c_buffer_r, data_masked, window_write
    data_write = data_masked[
                 r_buffer_b: data_masked.shape[0] - r_buffer_t,
                 c_buffer_l: data_masked.shape[1] - c_buffer_r
                 ]
    return Tile(data=data_write, window=window_write)

from joblib import Parallel, delayed

tiles = Parallel(n_jobs=2)(delayed(_fill_tile)(r, c) for c in range(num_cols-1)
                 for r in range(num_rows-1))

# This is the multiprocess write loop
# i = 0
# for r in range(num_rows-1):
#     for c in range(num_cols-1):
#         ods.write_band(1, tiles[i].data, window=tiles[i].window)
#         i += 1


rc_tuples = [(r, c) for c in range(num_cols-1) for r in range(num_rows-1)]
this_rank_jobs = np.array_split(rc_tuples, size)[rank]

print(rank, this_rank_jobs)


def _mpi_helper(list_of_tuples):
    return [_fill_tile(* l) for l in list_of_tuples]


this_rank_tiles = _mpi_helper(this_rank_jobs)

all_tiles = comm.gather(this_rank_tiles)


# write
if rank == 0:
    for s in range(size):
        for tile in all_tiles[s]:
            print(tile)
            ods.write_band(1, tile.data, window=tile.window)

ods.close()