# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa

from lance.file import LanceFileReader, LanceFileWriter


class TqModel:
    """A class that represents a TurboQuant model.

    TurboQuant (Zandieh et al., ICLR 2026) is data-oblivious: the codebook
    is determined solely by ``(dimension, num_bits)``. Only the rotation matrix
    (seeded) needs to be stored. Training is near-instant (<1ms).

    Can be saved / loaded to checkpoint progress during distributed index builds.

    Parameters
    ----------
    rotation_matrix : pa.FixedSizeListArray
        The d×d random orthogonal matrix (Haar-distributed) stored as d rows
        of d floats. Generated via QR decomposition of a seeded Gaussian matrix.
    num_bits : int
        Bit-width per coordinate (1-8). Higher = better recall, more storage.
        4-bit is the recommended default: ~8x compression with ~95% recall@1.
    seed : int
        RNG seed used to generate the rotation matrix. Same seed always
        produces the same matrix, enabling reproducible distributed builds.
    dimension : int
        Vector dimension. Must be >= 3.

    Examples
    --------
    >>> from lance.indices import IndicesBuilder
    >>> builder = IndicesBuilder(ds, "vector")
    >>> tq = builder.train_tq(num_bits=4, seed=42)
    >>> tq.save("tq_model.lance")
    >>> tq_loaded = TqModel.load("tq_model.lance")

    Notes
    -----
    Unlike PQ which stores a learned codebook (potentially megabytes), TQ only
    stores the rotation matrix. The codebook is derived deterministically from
    ``(dimension, num_bits)`` at load time.

    Storage of the rotation matrix: d×d×4 bytes. For d=768: ~2.36 MB.
    For d=1536: ~9.4 MB. A future Hadamard rotation option would reduce
    this to just d/8 bytes (~96 bytes for d=768).
    """

    def __init__(
        self,
        rotation_matrix: pa.FixedSizeListArray,
        num_bits: int,
        seed: int,
        dimension: int,
    ):
        self.rotation_matrix = rotation_matrix
        """The rotation matrix (d x d), stored as FixedSizeList of d rows."""
        self.num_bits = num_bits
        """Bit-width per coordinate (1-8)."""
        self.seed = seed
        """RNG seed for rotation matrix reproducibility."""
        self.dimension = dimension
        """Vector dimension."""

    def save(self, uri: str):
        """
        Save the TQ model to a lance file.

        Parameters
        ----------

        uri: str
            The URI to save the model to.
        """
        with LanceFileWriter(
            uri,
            pa.schema(
                [pa.field("rotation_matrix", self.rotation_matrix.type)],
                metadata={
                    b"num_bits": str(self.num_bits).encode(),
                    b"seed": str(self.seed).encode(),
                    b"dimension": str(self.dimension).encode(),
                },
            ),
        ) as writer:
            batch = pa.table([self.rotation_matrix], names=["rotation_matrix"])
            writer.write_batch(batch)

    @classmethod
    def load(cls, uri: str):
        """
        Load a TQ model from a lance file.

        Parameters
        ----------

        uri: str
            The URI to load the model from.
        """
        reader = LanceFileReader(uri)
        num_rows = reader.metadata().num_rows
        metadata = reader.metadata().schema.metadata
        num_bits = int(metadata[b"num_bits"].decode())
        seed = int(metadata[b"seed"].decode())
        dimension = int(metadata[b"dimension"].decode())
        rotation_matrix = (
            reader.read_all(batch_size=num_rows)
            .to_table()
            .column("rotation_matrix")
            .chunk(0)
        )
        return cls(rotation_matrix, num_bits, seed, dimension)
