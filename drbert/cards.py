"""Model cards"""

import csv
import os
from typing import Optional


_rolodex_path = os.path.join(os.path.dirname(__file__), 'rolodex.csv')


class ModelCard:
  """ Model Information and Path container
  Holds information regarding the specified
  biomedical transformer model

  Attributes
  ----------
  -Public-
  name : str
    name of the model e.g. BioBERT
  weight_path : str
    path to find model either locally or from huggingface.co models
  citation : str
    full citation of the paper that introduced the model
  backend : str
    specifies PyTorch or TensorFlow
  transformer_architecture : str
    specifies the architecture of the transformer e.g. BERT, roBERTa, etc.
  pretrain_dataset : str
    dataset the model was pretrained on
  additional_train_notes : str
    important training information
  misc : str
    additional relevant information

  -Protected-
  _rolodex : str
    path to the rolodex.csv file containing all the model information

  Methods
  -------
  None

  """

  def __init__(self,
               model_name: str,
               local: bool=False,
               local_path: str=None
               ):
    """
    Parameters
    ----------
    model_name : str
      name of the model to load info for
    local : bool
      if model weights are saved locally/in drive
        (if the model you are trying to use can be found via the huggingface
         collection, set to False)
    local_path : str
      path to local model
    """
    # Validate path to model info file
    self._rolodex = _rolodex_path
    self._check_rolodex_path()
    self.available_models = self.__get_available_models()

    # Model information
    self.model_name = model_name
    self.name = None
    self.weight_path = None
    self.citation = None
    self.backend = None
    self.transformer_architecture = None
    self.pretrain_dataset = None
    self.additional_train_notes = None
    self.misc = None

    # Loads model card info
    if model_name:
      if local:
        msg = "If local=True, then a path to the model weights is required"
        assert local_path, msg
        # self.weight_path = local_path
        raise NotImplementedError("Forthcoming feature")
      else:
        self._find_model(model_name)

  def _check_rolodex_path(self):
    """Checks that the path to the rolodex.csv file is correct"""
    if not os.path.isfile(self._rolodex):
      raise ValueError("rolodex.csv file could not be found! Please check path.")

  def _find_model(self, model_name):
    """Loads the specified model information"""
    with open(self._rolodex, newline='') as f:
      reader = csv.reader(f, delimiter=',', quotechar='"')

      next(reader)  #skip header

      card = None
      for row in reader:
        # Once model is found, get data and stop search
        if row[0] == model_name:
          card = row
          break

      if not card:
        msg = f"Model {model_name} not found!\nAvailable models are: {self.available_models}"
        raise ValueError(msg)
    f.close()

    self.name = card[1]
    self.weight_path = card[2]
    self.citation = card[3]
    self.backend = card[4]
    self.transformer_architecture = card[5]
    self.pretrain_dataset = card[6]
    self.additional_train_notes = card[7]
    self.misc = card[8]

  def __get_available_models(self):
    """Retreives list of models in the rolodex"""
    models = []
    with open(self._rolodex, newline='') as f:
      reader = csv.reader(f, delimiter=',', quotechar='"')
      next(reader) # skip header

      for row in reader:
        models.append(row[0])
    f.close()
    return models

  def __repr__(self):
    msg = f"ModelCard(\n"
    msg += f"\tmodel={self.model_name},\n"
    msg += f"\tpath={self.weight_path},\n"
    msg += f"\tbackend={self.backend}"
    msg += "\n)"
    return msg

  def __str__(self):
    """TODO: Make pretty with formatting"""
    chunks = []
    chunks.append("="*25 + "MODEL CARD" + "="*25 )
    chunks.append("Model:  " + self.name)

    chunks.append("Citation:")
    cite = self.citation
    while len(cite) > 60:
        chunks.append(cite[:60])
        cite = cite[60:]
    chunks.append(cite)

    chunks.append("Architecture:  " + self.transformer_architecture)

    chunks.append("Pretraining Dataset:")
    pretrain = self.pretrain_dataset
    while len(pretrain) > 60:
        chunks.append(pretrain[:60])
        pretrain = pretrain[60:]
    chunks.append(pretrain)

    if self.additional_train_notes:
      chunks.append("Additional Notes:")
      notes = self.additional_train_notes
      while len(notes) > 60:
        chunks.append(notes[:60])
        notes = notes[60:]
      chunks.append(notes)

    if self.misc:
      chunks.append("Miscellaneous:")
      chunks.append(self.misc)

    chunks.append("="*60)

    return '\n'.join(chunks)
