import torch
import numpy as np

from fastai.vision.all import DataLoaders, L, Learner


def get_class_weights(dls: DataLoaders, 
					 type: str = 'IFW', 
					 axis: int|None = None, 
					 k: float = 1) -> torch.tensor:
	"""
	Computes class weights for a loss function to handle class imbalance.

	Args:
		dls (DataLoaders): The fastai DataLoaders object.
		type (str): The weighting strategy. Options: 'IFW', 'SIF', 'ENoS', 'MFB'.
		axis (int | None, optional): If specified, computes class weights for a particular axis in multi-task models.
			Use None for single-task models. Defaults to None.
		k (float): Hyperparameter for 'SIF' and 'ENoS' methods.

	Returns:
		torch.tensor: A tensor of weights for each class.
	"""	
	# Extract integer labels from the training dataset. The +1 is needed because axis 0 is typically the input (e.g., image)
	y = L(map(lambda x: int(x[axis+1]), dls.train_ds)) if axis is not None else L(map(lambda x: int(x[1]), dls.train_ds))
	# Number of samples for each class
	counts = np.bincount(y)
	# List of all classes names
	vocab = dls.vocab[axis] if axis is not None else dls.vocab

    # Ensure counts have the same size as vocab
	if len(counts) < len(vocab):
		counts = np.pad(counts, (0, len(vocab) - len(counts)))
	# Prevent usage when some classes are not present on the training set
	if np.any(counts == 0):
		missing_classes = [vocab[i] for i, count in enumerate(counts) if count == 0]
		raise ValueError(f"All classes must be present in the training set. Missing: {missing_classes}")
	
	match type:
		case 'IFW':
			# Inverse Frequency Weighting: w_c = N / (C * n_c)
			# Most standard
			weights = len(y) / (len(np.unique(y)) * counts)
		case 'SIF':
			# Smoothed Inverse Frequency: w_c = 1 / log(k + n_c)
			# Prevents excessively large weights
			# Typical k is ~1.02
			weights = 1 / np.log(k + counts)
		case 'ENoS':
			# Efective Number of Samples: w_c = (1 - k) / (1 - k ^ n_c)
			# Assumes that the marginal benefit of new data decreases with increasing number of samples for a class.
			# Typical k is between 0.9 and 0.9999
			weights = (1 - k) / (1 - k ** counts)
		case 'MFB':
			# Median Frequency Balancing: w_c = median_freq / freq_c
			# Prevents weights from becoming too large or too small
			freq = counts / len(y)
			weights = np.median(freq) / freq
		case _:
			raise ValueError(f"'{type}' is not a valid type option.")
	print({f'{c}: {w.item():.4f}' for c,w in zip(vocab, weights)})
	return torch.FloatTensor(weights).to(dls.device)


def set_controlled_oversampling(learn: Learner, col: str, heuristic_power: float = 0.5) -> Learner:
    """
    Modifies the training DataLoader to use controlled oversampling for each epoch.

    Args:
        learn (Learner): The fastai Learner.
        col (str): The column name in `learn.dls.items` containing the class labels.
        heuristic_power (float): The power for the weighting heuristic. 
                                 0.0 = no oversampling, 0.5 = sqrt, 1.0 = inverse frequency.
    """
    # Labels to use for oversampling
    labels = learn.dls.items[col]
    # Class weights using the heuristic: 1 / (n_c ^ power)
    class_counts = labels.value_counts()
    class_weights = 1 / (class_counts ** heuristic_power)
    # Map class weights to samples
    sample_weights = labels.map(class_weights).to_numpy()

    # Training set size
    train_ds_size = len(learn.dls.train_ds)
    # New `get_idxs` function
    def _oversampled_get_idxs():
        """Samples indices with replacement based on pre-calculated weights."""
        return np.random.choice(
            a=train_ds_size,       # Sample from indices 0 to N-1
            size=train_ds_size,    # Generate a full epoch's worth of indices
            replace=True,          # Oversampling requires replacement
            p=sample_weights / sample_weights.sum() # Probabilities must sum to 1
        ).tolist()
    # Monkey-patch the training dataloader's `get_idxs` method
    learn.dls.train.get_idxs = _oversampled_get_idxs
    print('Oversampling configured.')
    return learn