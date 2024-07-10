import torch
from torch.utils import data

# dataset generator
class XorDataset(data.Dataset):
    """Create dataset for Xor learning."""

    def __init__(self, nsample=1000, test=False, discrete=True, range=(0,1)):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        # self.input_vars = torch.rand(self.nsample, 2)
        a, b = range
        if test:
            if discrete:
                self.input_vars = torch.tensor(
                    [[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float)
                self.nsample = 4
            else:
                self.nsample //= 10
                self.input_vars = torch.rand((self.nsample, 2)) * (b - a) + a
        else:
            if discrete:
                self.input_vars = torch.bernoulli(
                    torch.ones((self.nsample, 2)) * 0.5)
            else:
                self.input_vars = torch.rand((self.nsample, 2)) * (b - a) + a

    def __getitem__(self, index):
        """Get a data point."""
        assert index < self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, torch.logical_xor(*torch.round(inp)).type(torch.LongTensor)

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample


class OrDataset(data.Dataset):
    """Create dataset for Or learning."""

    def __init__(self, nsample=1000, test=False):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        # self.input_vars = torch.rand(self.nsample, 2)
        if test:
            self.input_vars = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float)
            self.nsample = 4
        else:
            self.input_vars = torch.bernoulli(
                torch.ones((self.nsample, 2)) * 0.5)

    def __getitem__(self, index):
        """Get a data point."""
        assert index < self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, torch.logical_or(*torch.round(inp)).type(torch.float)

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample