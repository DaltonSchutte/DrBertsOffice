import os

from google.colab import drive

class ColabLink:
  """Links Colab Notebook to Google Drive
  Provides collected interface to connect to your drive,
  load data, save data, and interact with the specified
  directory

  Attributes
  ----------
  base_dir : str
    path to the base directory
  paths : dict
    dictionary containing keys for user added paths

  Methods
  -------
  _connect_to_drive()
    Connects to the user's Google Drive
  _set_working_dir()
    Sets working directory to the directory provided at object
      instantiation
  add_path(path: str, name: str, overwrite: bool=False)
    adds a path to the paths dictionary
  list(name: str)
    lists the contents of the directory at paths[name]
  """

  def __init__(self,
               base_dir: str,
               ):
    """
    Parameters
    ----------
    base_dir : str
      directory to set for the base working directory
    drive : bool
      is the directory in Google Drive
    """
    self.base_dir = base_dir

    self._connect_to_drive()
    self._set_working_dir()

    self.paths = {'work': self.__working_dir}

  def __call__(self, name):
    if name not in self.paths.keys():
      raise ValueError(f"{name} not in the set of paths")
    return self.paths[name]

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    text = "KEY   |  PATH\n"
    text += f"work  |  {self.paths['work']}\n"
    for k, v in self.paths.items():
      if k != 'work':
        text += f"{k} |  {v}\n"

    return text

  def _connect_to_drive(self):
    """"Attempts the initial connection to gdrive"""
    print("Attempting to connect to gdrive...")
    drive.mount('/content/gdrive')
    print("[Connected to gdrive]")

  def _set_working_dir(self):
    """Sets the working directory"""
    print("Setting working directory...")
    dir_path = os.path.join('/content/gdrive', self.base_dir)

    self.__check_path(dir_path)

    self.__working_dir = dir_path
    os.chdir(self.__working_dir)
    print(f"Working directory set to {self.__working_dir}")
    print(f"Working directory can be accessed using 'work' in class call")
    print("[Working directory set]")

  def add_path(self,
               path: str,
               name: str,
               overwrite: bool=False
               ):
    """Add a path for quick access
    Parameters
    ----------
    path : str
      path to add to the quick lookup
    name : str
      lookup name for path
    overwrite : bool
      overwrite existing key
    """
    msg = "Trying to overwrite base directory key. Please change name"
    assert name != 'work', msg
    new_path = os.path.join(self.__working_dir, path)
    self.__check_path(new_path)

    if name in self.paths.keys():
      if overwrite:
        print(f"Overwriting {name} to new path...")
      else:
        msg = f"{name} is already a key in stored paths."
        msg += "Try another or set overwrite=True"
        raise ValueError(msg)

    self.paths.update({name: new_path})
    print(f"{name} set to {new_path}")
    print("[Path added]")

  def __check_path(self, path: str):
    """Validates the provided path"""
    is_dir = os.path.isdir(path)
    is_file = os.path.isfile(path)

    if (not is_dir) and (not is_file):
      raise OSError(f"Provided path {path} is not a valid file or directory")

  def __getitem__(self, key):
    return self.paths[key]

  def list(self, name: str=None):
    """Lists the contents of the specified directory"""
    if name:
      return os.listdir(self.paths[name])
    return os.listdir(self.paths['work'])
