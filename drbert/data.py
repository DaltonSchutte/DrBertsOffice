from torch.utils.data import Dataset

class TaskDataset(Dataset):
  """Custom dataset
  Holds inputs that are of the form:
  (TEXT)[SEP]subject[SEP]predicate[SEP]object
  Experiment expects 0 or 1 as a label
  #TODO: make a special dataset class for NER

  Attributes
  ----------
  None

  Methods
  -------
  None
  """
  
  def __init__(self, inputs, labels):
    self._inputs = inputs
    self._labels = labels

  def __len__(self):
    return len(self._labels)

  def __getitem__(self, idx):
    item = {k: v[idx].clone().detach() for k, v in self._inputs.items()}
    item['labels'] = torch.tensor(self._labels[idx])
    return item

from torch.utils.data import Dataset

class NLPDataset(Dataset):
  """Custom dataset
  Holds inputs that are of the form:
  (TEXT)[SEP]subject[SEP]predicate[SEP]object
  Experiment expects 0 or 1 as a label
  #TODO: make a special dataset class for NER

  Attributes
  ----------
  None

  Methods
  -------
  None
  """

  def __init__(self, inputs, labels):
    self._inputs = inputs
    self._labels = labels

  def __len__(self):
    return len(self._labels)

  def __getitem__(self, idx):
    item = {k: v[idx].clone().detach() for k, v in self._inputs.items()}
    item['labels'] = torch.tensor(self._labels[idx])
    return item
