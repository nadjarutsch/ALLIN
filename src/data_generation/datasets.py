import torch


class InterventionalDataset(object):
    """Dataset for interventional data from one or more intervened variables.

    Attributes:
        data_loaders: A dictionary containing one dataloader for each intervened variable.
        data_iter: A dictionary containing one iterable for each dataloader.
    """

    def __init__(self, dataloaders: dict):
        """Initializes the instance from given data.

        Args:
            dataloaders: Dataloaders that contain interventional data.
        """

        self.data_loaders = dataloaders
        self.data_iter = {}

        for var_idx in dataloaders.keys():
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])

    def get_batch(self, var_idx: int) -> torch.Tensor:
        """Samples a batch of interventional data for the specified intervention variable.

        Args:
            var_idx: Index of the intervened variable.

        Returns:
            A batch of interventional data, size batch_size x D.
        """

        try:
            batch = next(self.data_iter[var_idx])
        except StopIteration:
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])
            batch = next(self.data_iter[var_idx])
        return batch[0]
