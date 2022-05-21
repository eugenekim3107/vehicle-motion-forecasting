from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

ROOT_PATH = "./argo2/"

cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc",
          "palo-alto"]
splits = ["train", "test"]


def get_city_trajectories(city="palo-alto", split="train", normalized=False):
    f_in = ROOT_PATH + split + "/" + city + "_inputs"
    inputs = pickle.load(open(f_in, "rb"))
    inputs = np.asarray(inputs)

    outputs = None

    if split == "train":
        f_out = ROOT_PATH + split + "/" + city + "_outputs"
        outputs = pickle.load(open(f_out, "rb"))
        outputs = np.asarray(outputs)

    return inputs, outputs


class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""

    def __init__(self, city: str, split: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.transform = transform

        inputs, outputs = get_city_trajectories(city=city, split=split,
                                                normalized=False)
        comb = np.concatenate((inputs, outputs), axis=1).T
        test = np.tile(comb, 60).T
        final = np.empty((0, 51, 2))
        for i in range(60):
            final = np.concatenate(
                (final, test[0:inputs.shape[0], i:51 + i, :]))
        final = np.hsplit(final, [50])
        self.inputs = final[0]
        self.outputs = final[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        data = (self.inputs[idx], self.outputs[idx])

        if self.transform:
            data = self.transform(data)

        return data


# intialize a dataset
pa = 'palo-alto'
au = 'austin'
mi = 'miami'
pi = 'pittsburgh'
db = 'dearborn'
dc = 'washington-dc'
split = 'train'

train_dataset_pa = ArgoverseDataset(city=pa, split=split)
train_dataset_au = ArgoverseDataset(city = au, split = split)
train_dataset_pi = ArgoverseDataset(city = pi, split = split)
train_dataset_db = ArgoverseDataset(city = db, split = split)
train_dataset_dc = ArgoverseDataset(city = dc, split = split)
train_dataset_mi = ArgoverseDataset(city = mi, split = split)

bs = len(train_dataset_pa)
train_loader_pa = DataLoader(train_dataset_pa,batch_size=bs)
