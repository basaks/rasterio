import sys
from pathlib import Path
import numpy as np
from collections import namedtuple
import rasterio as rio
from rasterio.fill import fillnodata
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def _get_source_data(src, dest):

    ids = rio.open(src, 'r')
    data = None

    if rank == 0:
        if dest.exists():
            raise FileExistsError('Output file {} exists'.format(dest.as_posix()))
        else:  # copy
            dest.write_bytes(src.read_bytes())
        data = ids.read(1, masked=True)

        #  write nans in dest
        ods = rio.open(dest, 'r+')
        data_type = data.dtype
        ods.write_band(1, np.empty_like(data, dtype=data_type))
        ods.close()

    data = comm.bcast(data, root=0)  # read and bcast

    return data


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
    tile_data = data[window_read[0][0]:window_read[0][1],
                window_read[1][0]:window_read[1][1]]

    orig_data = data[rows[r]: rows[r+1], cols[c]: cols[c+1]]

    if tile_data.count() == tile_data.size:  # all unmasked pixels, nothing to do
        return Tile(data=orig_data, window=window_write)
    elif tile_data.count() == 0:  # all masked pixels, can't do filling
        return Tile(data=orig_data, window=window_write)

    data_masked = fillnodata(tile_data, mask=None, max_search_distance=kernel_size)
    print(r_buffer_b, r_buffer_t, c_buffer_l, c_buffer_r)

    # return r_buffer_b, r_buffer_t, c_buffer_l, c_buffer_r, data_masked, window_write
    data_write = data_masked[
                 r_buffer_b: data_masked.shape[0] - r_buffer_t,
                 c_buffer_l: data_masked.shape[1] - c_buffer_r
                 ]
    return Tile(data=data_write, window=window_write)


def _multiprocess(dest):
    from joblib import Parallel, delayed
    tiles = Parallel(n_jobs=2)(delayed(_fill_tile)(r, c) for c in range(num_cols-1)
                               for r in range(num_rows-1))

    # This is the multiprocess write loop
    i = 0
    ods = rio.open(dest, 'r+')
    for r in range(num_rows-1):
        for c in range(num_cols-1):
            ods.write_band(1, tiles[i].data, window=tiles[i].window)
            i += 1
    ods.close()


def _mpi_helper(list_of_tuples):
    return [_fill_tile(* l) for l in list_of_tuples]


if __name__ == '__main__':

    input_raster = sys.argv[1]
    kernel_size = int(sys.argv[2]) or 3
    num_rows = int(sys.argv[3]) or 10
    num_cols = int(sys.argv[4]) or 10

    src = Path(input_raster)
    dest = src.with_suffix('.filled.tif')
    data = _get_source_data(src, dest)

    rows = np.linspace(0, data.shape[0], num_rows, dtype=int)
    cols = np.linspace(0, data.shape[1], num_cols, dtype=int)
    Tile = namedtuple('Tile', ['data', 'window'])

    rc_tuples = [(r, c) for c in range(num_cols-1) for r in range(num_rows-1)]
    this_rank_jobs = np.array_split(rc_tuples, size)[rank]

    this_rank_tiles = _mpi_helper(this_rank_jobs)
    all_tiles = comm.gather(this_rank_tiles)

    # write filled data
    if rank == 0:
        ods = rio.open(dest, 'r+')
        for s in range(size):
            for tile in all_tiles[s]:
                ods.write_band(1, tile.data, window=tile.window)

        ods.close()
