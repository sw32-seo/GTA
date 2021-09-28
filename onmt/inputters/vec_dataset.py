import os

import torch
from torchtext.data import Field
import pickle

from onmt.inputters.datareader_base import DataReaderBase

try:
    import numpy as np
except ImportError:
    np = None


class VecDataReader(DataReaderBase):
    """Read feature vector data from disk.
    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing ``np`` fails.
    """

    def __init__(self):
        self._check_deps()

    @classmethod
    def _check_deps(cls):
        if np is None:
            cls._raise_missing_dep("np")

    def read(self, vecs, side, vec_dir=None):
        """Read data into dicts.
        Args:
            vecs (str or Iterable[str]): Sequence of feature vector paths or
                path to file containing feature vector paths.
                In either case, the filenames may be relative to ``vec_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            vec_dir (str): Location of source vectors. See ``vecs``.
        Yields:
            A dictionary containing feature vector data.
        """

        if isinstance(vecs, str):
            vecs = DataReaderBase._read_file(vecs)

        for i, vec in enumerate(vecs):
            yield {side: vec,
                   "indices": i}


def vec_sort_key(ex):
    """Sort using the length of the vector sequence."""
    return ex.src.shape[0]


class VecSeqField(Field):
    """Defines an vector datatype and instructions for converting to Tensor.
    See :class:`Fields` for attribute descriptions.
    """

    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=False, batch_first=False, pad_index=0,
                 is_target=False):
        super(VecSeqField, self).__init__(
            sequential=True, use_vocab=False, init_token=None,
            eos_token=None, fix_length=False, dtype=torch.float,
            preprocessing=preprocessing, postprocessing=postprocessing,
            lower=False, tokenize=None, include_lengths=include_lengths,
            batch_first=batch_first, pad_token=pad_index, unk_token=None,
            pad_first=False, truncate_first=False, stop_words=None,
            is_target=is_target
        )

    def pad(self, minibatch):
        """Pad a batch of examples to the length of the longest example.
        Args:
            minibatch (List[torch.FloatTensor]): A list of graph related mask data,
                each having shape ``(max_dim, max_dim)``
                where len is variable.
        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape
                ``(batch_size, max_dim, max_dim)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        assert not self.pad_first and not self.truncate_first \
            and not self.fix_length and self.sequential
        minibatch = list(minibatch)
        lengths_0 = [x.size(0) for x in minibatch]
        lengths_1 = [x.size(1) for x in minibatch]
        max_len_0 = max(lengths_0)
        max_len_1 = max(lengths_1)
        feats = torch.full((len(minibatch), max_len_0, max_len_1),
                           self.pad_token)
        for i, (feat, len_0, len_1) in enumerate(zip(minibatch, lengths_0, lengths_1)):
            feats[i, 0:len_0, 0:len_1] = feat
        if self.include_lengths:
            return (feats, (lengths_0, lengths_1))
        return feats

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.
        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): See `Field.numericalize`.
        """

        assert self.use_vocab is False
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)
        arr = arr.to(device)

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if self.sequential:
            arr = arr.contiguous()

        if self.include_lengths:
            return arr, lengths
        return arr


def vec_fields(**kwargs):
    vec = VecSeqField(pad_index=0, include_lengths=True)
    return vec
