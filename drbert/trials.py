import os
import json
from time import time

from tqdm import tqdm

import numpy as np
import scipy

from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AdamW, get_cosine_schedule_with_warmup


class Trials:
  """Class to handle the training and evaluation of a single BERT variant

  This class contains several functions to handle data splitting, model training,
  model evaluation, saving results, and saving the best model wrapped into a single
  method call to run_trials

  Attributes
  ----------
  # TODO: change appropriate objects to private attributes
  model
    class containing BERT model and tokenizer for a task
  data
    dictionary containing an entry for each of the train, dev, and test sets
  param
    dictionary containing experiment parameters
  results_dir
    location to save the results from the trials
  seeds
    array of random seeds for reproducability, also determines the number
    of trials that will be run
  device
    specifies if a gpu, tpu, or cpu will be used
  results
    dictionary where each entry corresponds to the results from a single trial
  optimizer
    PyTorch based optimization object
  scheduler
    learning rate scheduling object

  Methods
  -------
  run_trials
    begins the process of training and evaluating a model using each of the
    provided random seeds

  """

  def __init__(self,
               model,
               task: str,
               data: dict,
               param_dict: dict,
               results_dir: str,
               outfile: str,
               id_to_tag: dict,
               seeds: np.ndarray,
               device: str,
               ):
    """
    Parameters
    ----------
    model : Model
      Model object containing the pretrained BERT model and tokenizer with a new
      head inialized for a specific fine-tuning task
    data : dict
      dictionary containing special dataset objects for each of train, dev, and
      test sets
    param_dict : dict
      dictionary of specific experiment parameters
    results_dir : str
      location to save results
    seeds : np.ndarray
      array containing a unique random seed for each trial
    device : str
      specifies which accelerator hardware, if any, will be used
    """
    self.task = task
    self.data = data
    self.params = param_dict
    self.results_dir = results_dir
    self.seeds = seeds
    self.device = device
    self.outfile = outfile
    self.id_to_tag = id_to_tag

    self.results = {}
    self.model = model
    self.optimizer = None
    self.scheduler = None

    self.best_loss = np.inf

    self.train_loader = DataLoader(self.data['train'],
                                   batch_size=self.params['bsz'],
                                   shuffle=True)
    self.dev_loader = DataLoader(self.data['dev'],
                                   shuffle=True)
    self.test_loader = DataLoader(self.data['test'])

  def _reset_opts(self):
    """Resets the optimizer and learning rate scheduler"""
    self.optimizer = AdamW(self.model.parameters(),
                           lr=self.params['lr'],
                           weight_decay=self.params['wd']
                           )
    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                     num_warmup_steps=self.params['warmup'],
                                                     num_training_steps=self.params['epochs']*len(self.train_loader),
                                                    )

  def _reset_model(self):
    """Loads the model from the specified pretrained weights"""
    self.model._get_model()

  def _metrics(self, labels, preds):
    """Computes and returns metrics"""
    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()
    elif isinstance(labels, list):
      labels = np.array(labels)
    if isinstance(preds, torch.Tensor):
      preds = preds.numpy()
    elif isinstance(preds, list):
      preds = np.array(preds)

    idx = np.where(labels != -100)

    self.labels = labels[idx]
    self.preds = preds[idx]

    params = {'y_true': self.labels,
              'y_pred': self.preds,
              'zero_division': 0,
              'labels': [k for k, v in self.id_to_tag.items() if v != 'O']
              }

    recall = recall_score(average='micro', **params)
    precision = precision_score(average='micro', **params)
    f1 = f1_score(average='micro', **params)
    print(classification_report(target_names=[self.id_to_tag[k] for k in params['labels']],
                                **params))
    clf_report = classification_report(output_dict=True, **params)
    return recall, precision, f1, clf_report

  def _train(self):
    """One training epoch"""
    #############
    ### TRAIN ###
    #############
    train_loss = 0

    self.model.train()
    for i, items in enumerate(self.train_loader):
      items = {k: v.to(self.device) for k, v in items.items()}
      self.optimizer.zero_grad()
      outputs = self.model(**items)
      loss = outputs[0]
      loss.backward()
      nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
      self.optimizer.step()
      self.scheduler.step()

      train_loss += loss.item()

      if (i > 0) & ((i+1) % 25 == 0):
        print(f"\tBatch {i+1}: {round(train_loss/(i+1), 4)}")

    print("Train loss: {:.6f}".format(train_loss / len(self.train_loader)))

    ################
    ### VALIDATE ###
    ################
    dev_loss = 0
    dev_preds = []
    dev_labels = []

    self.model.eval()
    for item in self.dev_loader:
      dev_labels += item['labels'].detach().cpu().numpy().flatten().tolist()
      item = {k: v.to(self.device) for k, v in item.items()}
      outputs = self.model(**item)
      loss, logit = outputs[:2]

      if self.task == 'ner':
        d = 2
      else:
        d = 1

      pred = torch.argmax(F.softmax(logit, dim=d), dim=d)

      dev_loss += loss.cpu().item()
      dev_preds += pred.detach().cpu().numpy().flatten().tolist()

    rec, prec, f1, _ = self._metrics(dev_labels, dev_preds)
    print("Validation Results:\n",
          "Loss: {:.6f}\n".format(dev_loss/len(self.dev_loader))
          )

  def _eval(self):
    """Evaluation process"""
    test_loss = 0
    test_preds = []
    test_labels = []

    self.model.eval()
    for item in self.test_loader:
      test_labels += item['labels'].detach().cpu().numpy().flatten().tolist()
      item = {k: v.to(self.device) for k, v in item.items()}
      item.update({'return_dict': True})
      outputs = self.model(**item)
      loss, logit = outputs[:2]

      if self.task == 'ner':
        d = 2
      else:
        d = 1

      pred = torch.argmax(F.softmax(logit, dim=d), dim=d)
      test_loss += loss.cpu().item()
      test_preds += pred.detach().cpu().numpy().flatten().tolist()

    test_loss /= len(self.test_loader)
    rec, prec, f1, clf_report = self._metrics(test_labels, test_preds)
    print("Test Results:\n",
          "Mean pred value: {:.4f}\n".format(np.mean(test_preds)),
          "Loss: {:.6f}\n".format(test_loss)
          )

    return rec, prec, f1, clf_report, test_loss

  def _trial(self, num, seed):
    """Complete cycle of training and evaluation

    Parameters
    ----------
    num : int
      which trial this is in the sequence
    seed : int
      random seed to use for this trial
    """
    _start = time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    self._reset_model()
    self._reset_opts()

    self.model.to(self.device)

    for epoch in range(self.params['epochs']):
      print(f"Epoch: {epoch+1}")
      self._train()

    rec, prec, f1, clf_report, loss = self._eval()
    self.results[f'trial_{int(num)}'] = {'seed': int(seed),
                                    'recall': float(rec),
                                    'precision': float(prec),
                                    'f1': float(f1),
                                    'report': clf_report,
                                    'loss': float(loss),
                                    'runtime': time()-_start
                                    }

    if loss < self.best_loss:
      self.best_loss = loss
      self._save_model(num)


  def _save_model(self, num):
    """Saves the model at the specified path"""
    _path = self.results_dir+f'/trial_{int(num)}'
    if not os.path.isdir(_path):
      os.mkdir(_path)
    _path += f'/best_weights'
    # TODO: Add method to Model class
    self.model.bert.save_pretrained(_path)

  def _save_results(self):
    """Save the results in a json file"""
    self.results.update({'id_to_tag': self.id_to_tag})
    with open(os.path.join(self.results_dir, self.outfile+'.json'), 'w') as f:
      json.dump(self.results, f)
    f.close()

  def _print_stats(self, arr):
    n = len(arr)
    mean = np.mean(arr)
    sem = scipy.stats.sem(arr)
    dist = sem * scipy.stats.t.ppf((1+0.95)/2, n-1)
    print("{:.4f} ({:.4f}, {:.4f})".format(mean, mean-dist, mean+dist))

  def _summarize(self):
    recs = [v['recall'] for v in self.results.values()]
    precs = [v['precision'] for v in self.results.values()]
    f1s = [v['f1'] for v in self.results.values()]
    losses = [v['loss'] for v in self.results.values()]

    print("Loss Mean - 95% CI")
    self._print_stats(losses)
    print("\nRecall Mean - 95% CI")
    self._print_stats(recs)
    print("\nPrecision Mean - 95% CI")
    self._print_stats(precs)
    print("\nF1 Score Mean - 95% CI")
    self._print_stats(f1s)

  def _clean(self):
    self.model = None
    self.optimizer = None
    self.scheduler = None
    self.train_loader = None
    self.dev_loader = None
    self.test_loader = None

    torch.cuda.empty_cache()

  def run_trials(self):
    _start = time()
    for i, seed in tqdm(enumerate(self.seeds), total=len(self.seeds)):
      print()
      print(f"Trial {i+1}")
      self._trial(i+1, seed)
      print('\n\n')

    self.results.update({'total_runtime':time()-_start})
    self._save_results()
    self._summarize()
    self._clean()

