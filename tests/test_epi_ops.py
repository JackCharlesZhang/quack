import cutlass

from quack.epi_ops import ColVecReduce


class _ArgTensor:
    element_type = cutlass.Float32


def test_colvec_reduce_smem_bytes_default_to_direct_write_path():
    op = ColVecReduce("mColVecReduce")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == 0


def test_colvec_reduce_smem_bytes_use_atom_layout_n_warp_staging():
    op = ColVecReduce("mColVecReduce")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32), (4, 2, 1)) == 64 * 2 * 4
    assert op.smem_bytes(None, (64, 128, 64), (64, 32), (4, 2, 1)) == 0
