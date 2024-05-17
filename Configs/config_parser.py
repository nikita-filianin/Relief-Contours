import os
import yaml
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Config():
    def __init__(self, yml_path='config.yaml'):
        self.conf = AttrDict()
        with open(yml_path) as file:
            yaml_cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.conf.update(yaml_cfg)

    def getConfig(self, args: dict) -> dict:
        """
        Initializes some additional internal variables; based on user input in config.yaml
        """
        args = {key: value for key, value in args.items() if value != None}
        self.conf.update(args)

        # SET PATHS
        self.conf.JP2_DIR = f'{self.conf.STORAGE_PATH}/images/JP2'
        self.conf.JP2_FPATH = f'{self.conf.JP2_DIR}/{self.conf.imID}_RED.JP2'

        self.conf.PNG_DIR = f'{self.conf.STORAGE_PATH}/images/png'
        self.conf.PNG_FPATH = f'{self.conf.PNG_DIR}/{self.conf.imID}_RED.png'

        self.conf.RESULTS_DIR =f'{self.conf.STORAGE_PATH}/results/npy'
        self.conf.RESULTS_FPATH = f'{self.conf.RESULTS_DIR}/{self.conf.imID}.npy'

        self.conf.PLOT_DIR = f'{self.conf.STORAGE_PATH}/results/plots'
        self.conf.PLOT_FPATH = f'{self.conf.PLOT_DIR}/{self.conf.imID}.png'

        self.conf.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        return self.conf