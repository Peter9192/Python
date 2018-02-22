def weighted_mean(x,weights=None):
    if weights is None:
        weights = np.ones(x.shape)
    return np.sum(weights*x)/np.sum(weights)

def nanmean(x):
    x = x[~np.isnan(x)] # this flattens the array
    return np.sum(x)/float(len(x))

def weighted_nanmean(x,weights=None):
    if weights is None:
        weights = np.ones(x.shape)

    weights = weights[~np.isnan(x)]
    x = x[~np.isnan(x)] # x and weights still correspond

    return np.sum(weights*x)/np.sum(weights)

def weighted_nanmean(x,weights=None,axis=None):
    """ Apply weights along a specified axis before calculating the overall mean, while ignoring nans.

    Numpy's `nanmean` cannot deal with weights and numpy's `ma.average` can only calculate the weighted mean along a given axis.
    Note that `np.mean(np.nanmean(x,axis=0)) != np.mean(np.nanmean(x,axis=1))` for data with nans, because the not-nan elements in the corresponding rows/columns get relatively more weight.

    This function asserts that weights assumes the same shape as x before removing the nans and flattening the arrays in the process.
    """
    if weights is None:
        weights = np.ones(x.shape)
    weights = np.asarray(weights,dtype=float)

    if axis is None:
        assert weights.shape==x.shape, (
            "weights should have shape x or axis must be specified")
    else:
        assert len(weights.shape)==1 and len(weights)==x.shape[axis], (
            "weights should be 1d with same length as specified axis")

    if axis and len(x.shape)!=1:
        # e.g. if x.shape = (a,b,c,d) and axis = 2 weights is
        # reshaped to (1,1,c,1) and tiled along the other dims (a,b,1,d)
        weights_shape = np.ones(len(x.shape),dtype=int)
        weights_shape[axis] = x.shape[axis]
        tile_shape = list(x.shape)
        tile_shape[axis]=1
        weights = weights.reshape(weights_shape)
        weights = np.tile(weights,tile_shape)

    assert weights.shape==x.shape, "I messed up"

    # Remove nans from both x and weights
    weights = weights[~np.isnan(x)] # this flattens the array
    x = x[~np.isnan(x)] # x and weights still correspond

    return np.sum(weights*x)/np.sum(weights)

if __name__=="__main__":
    # Test array with weights
    x = np.arange(12,dtype=float).reshape(2,3,2)
    x[1,1,1] = np.nan
    y = np.arange(3)+1

    # Verifcation array
    x2 = np.array([[[0,1],[2,3],[2,3],[4,5],[4,5],[4,5]],
                   [[6,7],[8,np.nan],[8,np.nan],[10,11],[10,11],[10,11]]],dtype=float)
    a weighted_nanmean(x,weights=y,axis=1) == np.nanmean(x2)
