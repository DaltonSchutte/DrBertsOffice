from transformers import (
    BertForMultipleChoice, 
    BertForQuestionAnswering,
    BertForTokenClassification, 
    BertForSequenceClassification,
    BertTokenizerFast
)
                          

class Model:
  """Container for BERT model and tokenizer

  Attributes
  ----------
  weight_path : str
    path to the BERT weights (via huggingface or local)
  model_backend : str
    DL framework the model will load in naturally as (PyTorch or TensorFlow)
  desired_backend : str
    DL framework the user would like the model to use (PyTorch or TensorFlow)
  task : str
    tasl
  num_labels : int
    number of classes being predicted over
  dropout : float=0.03
    dropout value for the BERT outputs before going to the head
  bert :   transformers.models.bert.modeling_bert
    Huggingface BERT model for specified task initiated 
    using BertFor{task}.from_pretrained({weight_path})
  tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
    Huggingface fast BERT tokenizer with additional tokens added as required

  Methods
  -------
  from_model_card
    loads model using a ModelCard

  to
    sends model to specified device {cpu, gpu, tpu}

  forward
    performs a forward pass using the instantiated BERT model

  train
    sets bert attribute to training mode, computes gradients on forward pass

  eval
    sets bert attribute to evaluation mode, does not compute gradients on
    forward pass
  
  _get_model
    initiates the appropriate BERT model based on the desired task

  _get_tokenizer
    initiates the appropriate tokenizer

  _convert_model
    NotImplemented, will convert between backends as desired

  

  """

  def __init__(self, 
               weight_path: str,
               model_backend: str,
               desired_backend: str,
               task: str,
               num_labels: int,
               dropout: float=0.03,
               **kwargs
               ):
    self.weight_path = weight_path
    self.model_backend = model_backend
    assert desired_backend in ['PyTorch', 'TensorFlow']
    self.desired_backend = desired_backend
    self.task = task
    self.num_labels = num_labels
    self.dropout = dropout
    self.kwargs = kwargs

    self._get_model()
    self._get_tokenizer()



  @classmethod
  def from_model_card(cls, 
                      model_card: ModelCard, 
                      desired_backend: str,
                      task: str,
                      num_labels: int,
                      dropout: float,
                      **kwargs
                      ):
    """Loads the model using the ModelCard"""
    
    return cls(model_card.weight_path,
               model_card.backend,
               desired_backend,
               task,
               num_labels,
               dropout,
               **kwargs
               )
  
  def _get_model(self):
    """Makes appropriate head for the specified task"""
    if self.task == 'ner':
      self.bert = BertForTokenClassification.from_pretrained(self.weight_path,
                                                             num_labels=self.num_labels)
    elif self.task == 'rel_ex':
      # TODO: Add special rel_ex for entity extraction
      self.bert = BertForSequenceClassification.from_pretrained(self.weight_path,
                                                                num_outputs=self.num_labels)
      self.bert.resize_token_embeddings(4)
    elif self.task == 'seq_clf':
      self.bert = BertForSequenceClassification.from_pretrained(self.weight_path,
                                                                num_outputs=self.num_labels)
    elif self.task == 'mc':
      self.bert =  BertForMultipleChoice.from_pretrained(self.weight_path,
                                                         num_outputs=self.num_labels)
    elif self.task == 'qa':
      self.bert = BertForQuestionAnswering.from_pretrained(self.weight_path,
                                                           num_labels=self.num_labels)
    else:
      raise NotImplementedError(f"{self.task} is not an implemented task, use ['ner', 'rel_ex', 'seq_clf', 'mc', 'qa']")

  def _get_tokenizer(self):
    if self.task != 'rel_ex':
      self.tokenizer = BertTokenizerFast.from_pretrained(self.weight_path)
    else:
      self.tokenizer = BertTokenizerFast.from_pretrained(self.weight_path,
                                                         additional_special_tokens=['<e1>',
                                                                                    '</e1>',
                                                                                    '<e2>',
                                                                                    '</e2>'
                                                                                    ]
                                                        )

  def __convert_model(self, 
                      model_card: ModelCard, 
                      desired_backend: str
                      ):
    raise NotImplementedError("Feature forthcoming...")

  def to(self, device):
    self.bert.to(device)

  def train(self):
    self.bert.train()

  def eval(self):
    self.bert.eval()

  def forward(self, x):
    return self.bert(x)

  def __call__(self, x):
    return self.forward(x)
