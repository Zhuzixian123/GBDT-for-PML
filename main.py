from Config import config
from 核心代码.Data_loader import data_loader
from 核心代码.Trainer import trainer

Config=config()
Data_loader=data_loader(config=Config)
Trainer = trainer(data_loader=Data_loader, config=Config)
Trainer.train()


