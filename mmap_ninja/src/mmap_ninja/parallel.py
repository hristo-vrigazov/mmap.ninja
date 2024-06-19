from enum import Enum
from functools import partial
from tqdm.auto import tqdm

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel, delayed = None, None
    HAS_JOBLIB = False
else:
    HAS_JOBLIB = True


class _Exhausted(Enum):
    exhausted = 'EXHAUSTED'


EXHAUSTED = _Exhausted.exhausted


class ParallelBatchCollector:
    _parallel: Parallel = None

    def __init__(self, indexable, batch_size, n_jobs=None, verbose=False, **kwargs):
        self.indexable, self._obj_length, self._num_batches = self.verify(indexable, batch_size)
        self.batch_size = batch_size

        self._pbar = self._init_pbar(verbose)
        self._parallel = self.begin(n_jobs, **kwargs)
        self._batch_num = 0
        self._exhausted = False

    @staticmethod
    def verify(indexable, batch_size):
        try:
            _ = indexable.__getitem__
        except AttributeError:
            if callable(indexable):
                indexable = _IndexableWrap(indexable)
            else:
                msg = 'indexable must implement __getitem__ or be callable and take one integer argument.'
                raise TypeError(msg)

        try:
            length = len(indexable)
        except TypeError:
            length = None
            num_batches = None
        else:
            num_batches = length // batch_size + (length % batch_size != 0)

        return indexable, length, num_batches

    @staticmethod
    def begin(n_jobs: int, **kwargs):
        if n_jobs in (None, 1):
            return
        elif not HAS_JOBLIB:
            msg = 'joblib is not installed. Install joblib or run with n_jobs=None to ignore parallelization.'
            raise ImportError(msg)

        _parallel = Parallel(n_jobs=n_jobs, **kwargs)
        _parallel.__enter__()
        return _parallel

    def batches(self):
        while not self.exhausted():
            yield self.collect_batch()

    def collect_batch(self):
        if self._parallel is None:
            batch = self._collect_no_parallel_batch()
        else:
            batch = self._collect_parallel_batch()

        self._update_pbar(batch)
        return batch

    def _init_pbar(self, verbose):
        if not verbose:
            return None
        return tqdm(total=self._obj_length)

    def _update_pbar(self, batch):
        if self._pbar is not None:
            self._pbar.update(len(batch))

    def _collect_no_parallel_batch(self):
        results = [_get_from_indexable(self.indexable, j) for j in self._rng()]

        if self.exhausted(results):
            results = [r for r in results if r is not EXHAUSTED]

        return results

    def _collect_parallel_batch(self):
        func = delayed(partial(_get_from_indexable, self.indexable))

        results = self._parallel(func(j) for j in self._rng())

        if self.exhausted(results):
            results = [r for r in results if r is not EXHAUSTED]
            self._parallel.__exit__(None, None, None)

        return results

    def exhausted(self, results=()):
        self._exhausted = (
                self._exhausted or
                any(r is EXHAUSTED for r in results) or
                self.completed_batches()
        )

        return self._exhausted

    def completed_batches(self):
        return self._num_batches is not None and self._batch_num == self._num_batches

    def _rng(self):
        start = self.batch_size * self._batch_num
        stop = self.batch_size * (1 + self._batch_num)

        self._batch_num += 1

        return range(start, stop)


class _IndexableWrap:
    def __init__(self, func):
        self._func = func

    def __getitem__(self, item):
        return self._func(item)

    @property
    def wrapped(self):
        return self._func


class _IndexableLengthWrap(_IndexableWrap):
    def __init__(self, func, length):
        super().__init__(func)
        self.length = length

    def __len__(self):
        return self.length


def make_indexable(func, length=None):
    if length is not None:
        return _IndexableLengthWrap(func, length)
    return _IndexableWrap(func)


def _get_from_indexable(indexable, item,):
    try:
        return indexable[item]
    except (IndexError, KeyError):
        return EXHAUSTED
