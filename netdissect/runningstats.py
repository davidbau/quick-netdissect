'''
Running statistics on the GPU using pytorch.

RunningTopK maintains top-k statistics for a set of channels in parallel.
RunningQuantile maintains (sampled) quantile statistics for a set of channels.
'''

import torch, math, numpy

class RunningTopK:
    '''
    A class to keep a running tally of the the top k values (and indexes)
    of any number of torch feature components.  Will work on the GPU if
    the data is on the GPU.

    This version flattens all arrays to avoid crashes.
    '''
    def __init__(self, k):
        self.k = k
        self.count = 0
        # This version flattens all data internally to 2-d tensors,
        # to avoid crashes with the current pytorch topk implementation.
        # The data is puffed back out to arbitrary tensor shapes on ouput.
        self.data_shape = None
        self.top_data = None
        self.top_index = None
        self.next = 0
        self.linear_index = 0
        self.perm = None

    def add(self, data):
        '''
        Adds a batch of data to be considered for the running top k.
        The zeroth dimension enumerates the observations.  All other
        dimensions enumerate different features.
        '''
        if self.top_data is None:
            # Allocation: allocate a buffer of size 5*k, at least 10, for each.
            self.data_shape = data.shape[1:]
            feature_size = int(numpy.prod(self.data_shape))
            self.top_data = torch.zeros(
                    feature_size, max(10, self.k * 5), out=data.new())
            self.top_index = self.top_data.clone().long()
            self.linear_index = 0 if len(data.shape) == 1 else torch.arange(
                feature_size, out=self.top_index.new()).mul_(
                        self.top_data.shape[-1])[:,None]
        size = data.shape[0]
        sk = min(size, self.k)
        if self.top_data.shape[-1] < self.next + sk:
            # Compression: if full, keep topk only.
            self.top_data[:,:self.k], self.top_index[:,:self.k] = (
                    self.result(sorted=False, flat=True))
            self.next = self.k
            free = self.top_data.shape[-1] - self.next
        # Pick: copy the top sk of the next batch into the buffer.
        # Currently strided topk is slow.  So we clone after transpose.
        # TODO: remove the clone() if it becomes faster.
        cdata = data.contiguous().view(size, -1).t().clone()
        td, ti = cdata.topk(sk, sorted=False)
        self.top_data[:,self.next:self.next+sk] = td
        self.top_index[:,self.next:self.next+sk] = (ti + self.count)
        self.next += sk
        self.count += size

    def result(self, sorted=True, flat=False):
        '''
        Returns top k data items and indexes in each dimension,
        with channels in the first dimension and k in the last dimension.
        '''
        k = min(self.k, self.next)
        # bti are top indexes relative to buffer array.
        td, bti = self.top_data[:,:self.next].topk(k, sorted=sorted)
        # we want to report top indexes globally, which is ti.
        ti = self.top_index.view(-1)[
                (bti + self.linear_index).view(-1)
                ].view(*bti.shape)
        if flat:
            return td, ti
        else:
            return (td.view(*(self.data_shape + (-1,))),
                    ti.view(*(self.data_shape + (-1,))))


class RunningQuantile:
    """
    Streaming randomized quantile computation for torch.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates (or old-style percentiles) can be read out using
    quantiles(q) or percentiles(p).

    Accuracy scales according to resolution: the default is to
    set resolution to be accurate to better than 0.1%,
    while limiting storage to about 50,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(self, depth=1, resolution=6 * 1024, buffersize=None,
            dtype=torch.float, device=None, seed=None, state=None):
        if state is not None:
            self.set_state_dict(state)
            return
        self.dtype = dtype
        self.device = torch.device('cpu') if device is None else device
        self.resolution = resolution
        self.depth = depth
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = [torch.zeros(depth, resolution, dtype=dtype, device=device)]
        self.firstfree = [0]
        self.randbits = torch.ByteTensor(resolution)
        self.currentbit = len(self.randbits) - 1
        self.extremes = torch.zeros(depth, 2, dtype=dtype, device=device)
        self.extremes[:,0] = float('inf')
        self.extremes[:,-1] = -float('inf')
        self.size = 0

    def cuda(self):
        """Switches internal storage to cuda mode."""
        self.data = [d.cuda() for d in self.data]
        self.extremes = self.extremes.cuda()
        self.device = self.extremes.device

    def cpu(self):
        """Switches internal storage to cpu mode."""
        self.data = [d.cpu() for d in self.data]
        self.extremes = self.extremes.cpu()
        self.device = self.extremes.device

    def add(self, incoming):
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth
        self.size += incoming.shape[0]
        # Convert to a flat torch array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = int(math.ceil(self.buffersize / self.samplerate))
        for index in range(0, len(incoming), chunksize):
            batch = incoming[index:index+chunksize]
            sample = sample_portion(batch, self.samplerate)
            if len(sample):
                self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        # First time sampling - the data source is very large.
                        self._scan_extremes(incoming)
                    incoming = sample_portion(incoming, self.samplerate)
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:,ff:ff + copycount] = torch.t(
                    incoming[index:index + copycount,:])
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
                -(-self.data[index-1].shape[1] // 2) if index else 1):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:,0:self.firstfree[index]]
            data = data.sort()[0]
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:,0], data[:,-1])
            offset = self._randbit()
            position = self.firstfree[index + 1]
            subset = data[:,offset::2]
            self.data[index + 1][:,position:position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1
        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
                torch.min(incoming, dim=0)[0],
                torch.max(incoming, dim=0)[0])

    def _update_extremes(self, minr, maxr):
        self.extremes[:,0] = torch.min(
                torch.stack([self.extremes[:,0], minr]), dim=0)[0]
        self.extremes[:,-1] = torch.max(
                torch.stack([self.extremes[:,-1], maxr]), dim=0)[0]

    def _randbit(self):
        self.currentbit += 1
        if self.currentbit >= len(self.randbits):
            self.randbits.random_(to=2)
            self.currentbit = 0
        return self.randbits[self.currentbit]

    def state_dict(self):
        return dict(
                resolution=self.resolution,
                depth=self.depth,
                buffersize=self.buffersize,
                samplerate=self.samplerate,
                data=[d.cpu().numpy()[:,:f].T
                    for d, f in zip(self.data, self.firstfree)],
                sizes=[d.shape[1] for d in self.data],
                extremes=self.extremes.cpu().numpy(),
                size=self.size)

    def set_state_dict(self, dic):
        self.resolution = int(dic['resolution'])
        self.randbits = torch.ByteTensor(self.resolution)
        self.currentbit = len(self.randbits) - 1
        self.depth = int(dic['depth'])
        self.buffersize = int(dic['buffersize'])
        self.samplerate = float(dic['samplerate'])
        firstfree = []
        buffers = []
        for d, s in zip(dic['data'], dic['sizes']):
            firstfree.append(d.shape[0])
            buf = numpy.zeros((d.shape[1], s), dtype=d.dtype)
            buf[:,:d.shape[0]] = d.T
            buffers.append(torch.from_numpy(buf))
        self.firstfree = firstfree
        self.data = buffers
        self.extremes = torch.from_numpy((dic['extremes']))
        self.size = int(dic['size'])
        self.dtype = self.extremes.dtype
        self.device = self.extremes.device

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:,:self.firstfree[0]].t())
        return self.extremes.clone()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(0, torch.zeros(self.depth, cap,
                dtype=self.dtype, device=self.device))
            self.firstfree.insert(0, 0)
        else:
            # Unless we're so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index-1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level's buffer size (rounding up)
            if self.data[index-1].shape[1] - (amount + position) >= (
                    -(-self.data[index-2].shape[1] // 2) if (index-1) else 1):
                self.data[index-1][:,position:position + amount] = (
                        self.data[index][:,:amount])
                self.firstfree[index-1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:,:amount]
                data = data.sort()[0]
                if index == 1:
                    self._update_extremes(data[:,0], data[:,-1])
                offset = self._randbit()
                scrunched = data[:,offset::2]
                self.data[index][:,:scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = int(math.ceil(self.resolution * (0.67 ** len(self.data))))
        if cap < 2:
            return 0
        # Round up to the nearest multiple of 8 for better GPU alignment.
        cap = -8 * (-cap // 8)
        return max(self.buffersize, cap)

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:,:self.firstfree[0]].t())
        size = sum(self.firstfree) + 2
        weights = torch.FloatTensor(size) # Floating point
        summary = torch.zeros(self.depth, size,
                dtype=self.dtype, device=self.device)
        weights[0:2] = 0
        summary[:,0:2] = self.extremes
        index = 2
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:,index:index + ff] = self.data[level][:,:ff]
            weights[index:index + ff] = 2.0 ** level
            index += ff
        assert index == summary.shape[1]
        if sort:
            summary, order = torch.sort(summary, dim=-1)
            weights = weights[order.view(-1).cpu()].view(order.shape)
        return (summary, weights)

    def quantiles(self, quantiles, old_style=False):
        if self.size == 0:
            return torch.full((self.depth, len(quantiles)), torch.nan)
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        if old_style:
            # To be convenient with torch.percentile
            cumweights -= cumweights[:,0:1].clone()
            cumweights /= cumweights[:,-1:].clone()
        else:
            cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros(self.depth, len(quantiles),
                dtype=self.dtype, device=self.device)
        # numpy is needed for interpolation
        if not hasattr(quantiles, 'cpu'):
            quantiles = torch.Tensor(quantiles)
        nq = quantiles.cpu().numpy()
        ncw = cumweights.cpu().numpy()
        nsm = summary.cpu().numpy()
        for d in range(self.depth):
            result[d] = torch.tensor(numpy.interp(nq, ncw[d], nsm[d]),
                    dtype=self.dtype, device=self.device)
        return result

    def integrate(self, fun):
        result = None
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            term = torch.sum(
                    fun(self.data[level][:,:ff]) * (2.0 ** level),
                    dim=-1)
            if result is None:
                result = term
            else:
                result += term
        if result is not None:
            result /= self.samplerate
        return result

    def percentiles(self, percentiles):
        return self.quantiles(percentiles, old_style=True)

    def readout(self, count=1001, old_style=True):
        return self.quantiles(
                torch.linspace(0.0, 1.0, count), old_style=old_style)

def sample_portion(vec, p=0.5):
    bits = torch.bernoulli(torch.zeros(vec.shape[0], dtype=torch.uint8,
        device=vec.device), p)
    return vec[bits]

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("error")
    import time
    import argparse
    parser = argparse.ArgumentParser(
        description='Test things out')
    parser.add_argument('--mode', default='cpu', help='cpu or cuda')
    parser.add_argument('--test_size', type=int, default=1000000)
    args = parser.parse_args()

    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    amount = args.test_size
    quantiles = 1000
    data = numpy.arange(float(amount))
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    depth = 50
    test_cuda = torch.cuda.is_available()
    alldata = data[:,None] + (numpy.arange(depth) * amount)[None, :]
    actual_sum = torch.FloatTensor(numpy.sum(alldata * alldata, axis=0))
    amt = amount // depth
    for r in range(depth):
        numpy.random.shuffle(alldata[r*amt:r*amt+amt,r])
    if args.mode == 'cuda':
        alldata = torch.cuda.FloatTensor(alldata)
        dtype = torch.float
        device = torch.device('cuda')
    else:
        alldata = torch.FloatTensor(alldata)
        dtype = torch.float
        device = None
    starttime = time.time()
    qc = RunningQuantile(depth=depth, dtype=dtype, device=device,
            resolution=6 * 1024)
    qc.add(alldata)
    # Test state dict
    saved = qc.state_dict()
    # numpy.savez('foo.npz', **saved)
    # saved = numpy.load('foo.npz')
    qc = RunningQuantile(state=saved)
    assert not qc.device.type == 'cuda'
    qc.add(alldata)
    actual_sum *= 2
    ro = qc.readout(1001).cpu()
    endtime = time.time()
    gt = torch.linspace(0, amount, quantiles+1)[None,:] + (
            torch.arange(qc.depth, dtype=torch.float) * amount)[:,None]
    maxreldev = torch.max(torch.abs(ro - gt) / amount) * quantiles
    print("Maximum relative deviation among %d perentiles: %f" % (
        quantiles, maxreldev))
    minerr = torch.max(torch.abs(qc.minmax().cpu()[:,0] -
            torch.arange(qc.depth, dtype=torch.float) * amount))
    maxerr = torch.max(torch.abs((qc.minmax().cpu()[:, -1] + 1) -
            (torch.arange(qc.depth, dtype=torch.float) + 1) * amount))
    print("Minmax error %f, %f" % (minerr, maxerr))
    interr = torch.max(torch.abs(qc.integrate(lambda x: x * x).cpu()
            - actual_sum) / actual_sum)
    print("Integral error: %f" % interr)
    counterr = ((qc.integrate(lambda x: torch.ones(x.shape[-1]).cpu())
                - qc.size) / (0.0 + qc.size)).item()
    print("Count error: %f" % counterr)
    print("Time %f" % (endtime - starttime))
    # Algorithm is randomized, so some of these will fail with low probability.
    assert maxreldev < 1.0
    assert minerr == 0.0
    assert maxerr == 0.0
    assert interr < 0.01
    assert abs(counterr) < 0.001
    print("OK")

