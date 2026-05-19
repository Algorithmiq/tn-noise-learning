"""Custom quimb tensor-network class for 1-D LPDO super-operators.

An LPDO (Locally Purified Density Operator) stores a quantum channel via its
Kraus tensors plus a dual (conjugate) branch, so the resulting tensor network
represents a physically valid channel by construction.
"""

from __future__ import annotations

import quimb as qu
import quimb.tensor.array_ops as ops
from autoray import transpose
from quimb.tensor import TensorNetwork1D
from quimb.tensor.tensor_core import Tensor, rand_uuid, tags_to_oset


class LPDOSuperOperator1D(TensorNetwork1D):
    r"""1-D LPDO super-operator over a chain of qubits.

    Network shape::

        0   1   2       n-1
        |   |   |        |     <-- outer_upper_ind_id
        O===O===O==     =O
        |\  |\  |\       |\    <-- inner_upper_ind_id
          )   )   ) ...    )   <-- K (size of local Kraus sum)
        |/  |/  |/       |/    <-- inner_lower_ind_id
        O===O===O==     =O
        |   | : |        |     <-- outer_lower_ind_id
              :
             chi (virtual / entangling bond dim)

    Args:
        arrays: sequence of 2n arrays — the first two are the upper and lower
            tensors on site 0, etc. Arrays are 5-D for cyclic boundary
            conditions, and 4-D at the chain ends for open boundaries.
        shape: ordering string for the tensor axes (defaults to ``"lrkud"``).
        site_tag_id: format string for per-site tags.
        outer_upper_ind_id / inner_upper_ind_id / inner_lower_ind_id /
        outer_lower_ind_id: format strings for the four index families shown
            in the diagram.
        tags / tags_upper / tags_lower: extra tags applied to all tensors,
            the upper branch only, and the lower branch only.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_outer_upper_ind_id",
        "_inner_upper_ind_id",
        "_inner_lower_ind_id",
        "_outer_lower_ind_id",
        "cyclic",
        "_L",
    )

    def __init__(
        self,
        arrays,
        shape="lrkud",
        site_tag_id="I{}",
        outer_upper_ind_id="kn{}",
        inner_upper_ind_id="k{}",
        inner_lower_ind_id="b{}",
        outer_lower_ind_id="bn{}",
        tags=None,
        tags_upper=None,
        tags_lower=None,
        **tn_opts,
    ):
        if isinstance(arrays, LPDOSuperOperator1D):
            super().__init__(arrays)
            return

        arrays = tuple(arrays)
        self._L = len(arrays) // 2

        self._outer_upper_ind_id = outer_upper_ind_id
        self._inner_upper_ind_id = inner_upper_ind_id
        self._inner_lower_ind_id = inner_lower_ind_id
        self._outer_lower_ind_id = outer_lower_ind_id

        self._site_tag_id = site_tag_id
        tags = tuple(tags_to_oset(tags))
        tags_upper = tuple(tags_to_oset(tags_upper))
        tags_lower = tuple(tags_to_oset(tags_lower))

        sites_present = tuple(self.sites)
        outer_upper_inds = map(outer_upper_ind_id.format, sites_present)
        inner_upper_inds = map(inner_upper_ind_id.format, sites_present)
        inner_lower_inds = map(inner_lower_ind_id.format, sites_present)
        outer_lower_inds = map(outer_lower_ind_id.format, sites_present)

        def gen_tags():
            for site_tag in self.site_tags:
                yield (site_tag, "V") + tags + tags_upper
                yield (site_tag, "Vdag") + tags + tags_lower

        self.cyclic = ops.ndim(arrays[0]) == 5

        # Transpose arrays to 'lrkud' order:
        #        u
        #        |
        #     l--O--r
        #        |\
        #        d k
        def gen_orders():
            lkud_ord = tuple(shape.replace("r", "").find(x) for x in "lkud")
            rkud_ord = tuple(shape.replace("l", "").find(x) for x in "rkud")
            lrkud_ord = tuple(map(shape.find, "lrkud"))
            yield rkud_ord if not self.cyclic else lrkud_ord
            yield rkud_ord if not self.cyclic else lrkud_ord
            for _ in range(self.L - 2):
                yield lrkud_ord
                yield lrkud_ord
            yield lkud_ord if not self.cyclic else lrkud_ord
            yield lkud_ord if not self.cyclic else lrkud_ord

        def gen_inds():
            if self.cyclic:
                cycU_ix, cycL_ix = (rand_uuid(),), (rand_uuid(),)
            else:
                cycU_ix, cycL_ix = (), ()
            nU_ix, nL_ix, k_ix = rand_uuid(), rand_uuid(), rand_uuid()
            yield (*cycU_ix, nU_ix, k_ix, next(outer_upper_inds), next(inner_upper_inds))
            yield (*cycL_ix, nL_ix, k_ix, next(outer_lower_inds), next(inner_lower_inds))
            pU_ix, pL_ix = nU_ix, nL_ix
            for _ in range(self.L - 2):
                nU_ix, nL_ix, k_ix = rand_uuid(), rand_uuid(), rand_uuid()
                yield (pU_ix, nU_ix, k_ix, next(outer_upper_inds), next(inner_upper_inds))
                yield (pL_ix, nL_ix, k_ix, next(outer_lower_inds), next(inner_lower_inds))
                pU_ix, pL_ix = nU_ix, nL_ix
            k_ix = rand_uuid()
            yield (pU_ix, *cycU_ix, k_ix, next(outer_upper_inds), next(inner_upper_inds))
            yield (pL_ix, *cycL_ix, k_ix, next(outer_lower_inds), next(inner_lower_inds))

        def gen_tensors():
            for array, t_tags, inds, order in zip(arrays, gen_tags(), gen_inds(), gen_orders(), strict=False):
                yield Tensor(transpose(array, order), inds=inds, tags=t_tags)

        super().__init__(gen_tensors(), virtual=True, **tn_opts)

    @classmethod
    def rand(cls, n, K, chi, phys_dim=2, herm=True, cyclic=False, dtype=complex, **superop_opts):
        """Build a random LPDO with virtual bond χ and Kraus dimension K.

        Args:
            n: number of qubits.
            K: Kraus dimension (number of Kraus operators per site).
            chi: virtual bond dimension.
            phys_dim: physical dimension per site (2 for qubits).
            herm: if True, the lower branch is the complex conjugate of the
                upper branch (produces a physically valid Hermitian channel).
            cyclic: periodic boundary conditions.
            dtype: numpy dtype of the random tensors.
            **superop_opts: forwarded to the class constructor.
        """

        def gen_arrays():
            for i in range(n):
                shape = []
                if cyclic or (i != 0):
                    shape += [chi]
                if cyclic or (i != n - 1):
                    shape += [chi]
                shape += [K, phys_dim, phys_dim]
                data = qu.randn(shape=shape, dtype=dtype)
                yield data
                if herm:
                    yield data.conj()
                else:
                    yield qu.randn(shape=shape, dtype=dtype)

        arrays = map(ops.sensibly_scale, gen_arrays())
        return cls(arrays, **superop_opts)

    @property
    def outer_upper_ind_id(self):
        """Format string for the outer-upper indices (one per site)."""
        return self._outer_upper_ind_id

    @property
    def inner_upper_ind_id(self):
        """Format string for the inner-upper indices (one per site)."""
        return self._inner_upper_ind_id

    @property
    def inner_lower_ind_id(self):
        """Format string for the inner-lower indices (one per site)."""
        return self._inner_lower_ind_id

    @property
    def outer_lower_ind_id(self):
        """Format string for the outer-lower indices (one per site)."""
        return self._outer_lower_ind_id
