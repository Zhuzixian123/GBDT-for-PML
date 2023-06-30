import os
import pickle


class saver:
    def __init__(self,config):
        self.config=config

        self.best_loss=0

        pass

    def save_best_model(self,model):
        pass

    def save_last_model(self,model):

        model_path='./训练结果/'+self.config.train_params['model_name']+'/last_model.pkl'
        # 创建保存模型的文件夹（如果不存在）
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # 保存模型到文件
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        pass

    def load_model(self):

        model_path=self.config.train_params['model_path']
        # 加载模型
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

