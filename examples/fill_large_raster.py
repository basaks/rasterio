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

    # data = None

    if rank == 0:
        dest.write_bytes(src.read_bytes())
        ids = rio.open(src.as_posix(), 'r')
        # data = ids.read(1, masked=True)

        #  write nans in dest
        ods = rio.open(dest.as_posix(), 'r+')
        data_type = ids.dtypes[0]

        ods.write_band(1, np.empty(ids.shape, dtype=data_type))
        ods.close()
        ids.close()

    # can't bcast more than 4GB data
    # data = comm.bcast(data, root=0)  # read and bcast

    return None


def _fill_tile(r, c, row_min, row_max):
    # rows = np.linspace(0, ids.shape[0], num_rows, dtype=int)
    rows = np.linspace(row_min, row_max, num_rows, dtype=int)
    cols = np.linspace(0, ids.shape[1], num_cols, dtype=int)

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

    tile_data = ids.read(1, masked=True, window=window_read)

    print(tile_data.shape)

    if tile_data.count() == tile_data.size:  # all unmasked pixels, nothing to do
        orig_data = ids.read(1, masked=True, window=window_write)
        return Tile(data=orig_data, window=window_write)
    elif tile_data.count() == 0:  # all masked pixels, can't do filling
        orig_data = ids.read(1, masked=True, window=window_write)
        return Tile(data=orig_data, window=window_write)

    data_filled = fillnodata(tile_data, mask=None, max_search_distance=kernel_size)

    # return r_buffer_b, r_buffer_t, c_buffer_l, c_buffer_r, data_filled, window_write
    data_write = data_filled[
                 r_buffer_b: data_filled.shape[0] - r_buffer_t,
                 c_buffer_l: data_filled.shape[1] - c_buffer_r
                 ]
    return Tile(data=data_write, window=window_write)


def _multiprocess(dest, rows, cols):
    from joblib import Parallel, delayed
    tiles = Parallel(n_jobs=2)(delayed(_fill_tile)(r, c, rows, cols)
                               for c in range(num_cols-1)
                               for r in range(num_rows-1))

    # This is the multiprocess write loop
    i = 0
    ods = rio.open(dest.as_posix(), 'r+')
    for r in range(num_rows-1):
        for c in range(num_cols-1):
            ods.write_band(1, tiles[i].data, window=tiles[i].window)
            i += 1
    ods.close()


def _mpi_helper(list_of_tuples):
    return [_fill_tile(* l) for l in list_of_tuples]


Tile = namedtuple('Tile', ['data', 'window'])


def _fill_partition(r_min, r_max):

    rc_tuples = [(r, c, r_min, r_max)
                 for c in range(num_cols - 1)
                 for r in range(num_rows - 1)]
    this_rank_jobs = np.array_split(rc_tuples, size)[rank]
    this_rank_tiles = _mpi_helper(this_rank_jobs)
    all_tiles = comm.gather(this_rank_tiles)

    # write filled data
    if rank == 0:
        print('now writing data')
        ods = rio.open(dest.as_posix(), 'r+')
        for s in range(size):
            for tile in all_tiles[s]:
                ods.write_band(1, tile.data, window=tile.window)

        ods.close()


def test_fill_no_data(input_raster, filled_raster):
    """
    1. Test that unmasked pixels in the input data remain intact
    2. Test that tiling and partitioning has been implemented correctly.
    3. Test that if nodata is present in input raster, some are filled.

    :param input_raster: input raster before nodata fill
    :param filled_raster: output nodata filled raster

    """
    ids = rio.open(input_raster.as_posix(), 'r')
    idata = ids.read(1, masked=True)
    i_nodata = ids.get_nodatavals()[0]

    ods = rio.open(filled_raster.as_posix(), 'r')
    odata = ods.read(1, masked=True)

    # assign input nodatavalue in positions where input data was masked
    odata.data[idata.mask] = i_nodata

    # test that the tiles and partitions have been implemented correctly
    np.testing.assert_array_almost_equal(idata.data, odata.data)

    # test that if nodata holes exist in the input raster, they are filled
    if idata.mask.sum():
        assert idata.mask.sum() > odata.mask.sum()


if __name__ == '__main__':

    input_raster = sys.argv[1]
    kernel_size = int(sys.argv[2]) or 3
    num_rows = int(sys.argv[3]) + 1 or 10  # num rows per tile
    num_cols = int(sys.argv[4]) + 1 or 10  # num cols per tile
    partitions = int(sys.argv[5]) or 1

    src = Path(input_raster)
    dest = src.with_suffix('.filled.tif')

    # manage files
    _get_source_data(src, dest)

    ids = rio.open(src.as_posix(), 'r')

    p_rows_min_max = np.linspace(0, ids.shape[0], partitions + 1, dtype=int)

    for p in range(partitions):
        r_min = p_rows_min_max[p] if p == 0 else p_rows_min_max[p] - kernel_size
        r_max = p_rows_min_max[p+1] if p == partitions - 1 \
            else p_rows_min_max[p+1] + kernel_size
        print(r_min, r_max)
        _fill_partition(r_min, r_max)

    ids.close()

    test_fill_no_data(src, dest)
