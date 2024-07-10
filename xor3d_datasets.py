import torch
from torch.utils import data

# dataset generator
class Xor3dDataset(data.Dataset):
    """Create dataset for 3D Xor learning."""

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
                self.input_vars = torch.cartesian_prod(
                    torch.tensor([a,b]), torch.tensor([a,b]), torch.tensor([a,b]),
                    ).float()
                self.nsample = len(self.input_vars)
            else:
                self.nsample //= 10
                self.input_vars = torch.rand((self.nsample, 3)) * (b - a) + a
        else:
            if discrete:
                self.input_vars = torch.bernoulli(
                    torch.ones((self.nsample, 3)) * 0.5)
            else:
                self.input_vars = torch.rand((self.nsample, 3)) * (b - a) + a

    def __getitem__(self, index):
        """Get a data point."""
        assert index <= self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, torch.logical_xor(torch.round(inp[0]), torch.logical_xor(*torch.round(inp[1:]))).long() 

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample
