
import pandas as pd
from .Model import Line as model
from .Printer import run_printer
from .Saver import saver
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import time

class trainer:
    def __init__(self,data_loader,config):

        self.data_loader=data_loader
        self.config=config

        self.saver=saver(config)

        self.config1=self.config.config1

        self.device = self.config.base_params['device']
        self.lr=self.config.train_params['learning_rate']

        #将所有训练数据读取到内存中，不能读取到显存中（防止显存不足情况）
        self._load_dataset()

        #加载进度条
        self.run_printer=run_printer(self.data_shape)


        self.model=GradientBoostingRegressor(**self.config1,)

        if self.config.train_params['load_model']:
            self.model=self.saver.load_model()

    def train(self):

        if not self.config.train_params['load_model']:

            X=self.input_train.reshape((-1,13))
            Y=self.output_train.reshape(-1)

            first=time.time()
            self.model.fit(X,Y)
            last = time.time()
            print('数据尺寸：',X.shape,'训练用时：',last-first,'s')
        pre_loss=0
        pre_score=self._text_all()
        self.run_printer.print_result(pre_loss,pre_score)

    def _text_all(self):

        accumulated_loss = 0  # 用来记录loss

        #测试模型
        restored_data = np.zeros(self.output.shape).reshape(-1)
        for step in range(self.input.shape[0]):
            X, Y = self.input[step,:],self.output[step,:]
            restored_data[step]= self.model.predict(X.reshape(1, -1)).reshape(-1)


        data=self.output.reshape(-1)
        data_pre=restored_data.reshape(-1)

        up_1rx = np.sqrt(np.abs(np.sum(np.abs(np.square(data) - np.square(data_pre)))))
        down_1rx = np.sqrt(np.sum(np.square(data)))
        zhibiao_1r = up_1rx / down_1rx

        self.saver.save_last_model(self.model)

        # 定义一个空列表，用于保存每个迭代周期后的测试集均方根误差（RMSE）
        loss_scores = []
        # 迭代训练模型
        for i, y_pred in enumerate(self.model.staged_predict(self.data_loader.input)):
            # 计算当前迭代周期的均方根误差（RMSE）
            up_1 = np.sqrt(np.abs(np.sum(np.abs(np.square(self.data_loader.output.reshape(-1)) - np.square(y_pred)))))
            down_1 = np.sqrt(np.sum(np.square(self.data_loader.output.reshape(-1))))
            zhibiao_1 = up_1 / down_1
            loss_scores.append(1-zhibiao_1)
        df_in = pd.DataFrame(loss_scores)

        if self.config.train_params['save_result']:
            df_in.to_csv("./Result/" + "hz"+'_'+
                         str(self.config1['learning_rate'])+'_'+
                         str(self.config1['max_depth'])+'_'+
                         str(self.config1['min_samples_split'])+'_'+
                         str(self.config1['subsample'])+'_'+".csv", index=False,
                         header=False)


        aabc=self.data_loader.input

        first = time.time()
        data1111=self.model.predict(aabc)
        last = time.time()
        print('数据尺寸：',data1111.shape,'测试用时：', last - first,'s')


        return 1-zhibiao_1r

    def _load_dataset(self):
        use_data=self.config.train_params['use_data']
        if use_data=='train':
            self.input_train=self.data_loader.input_train
            self.output_train = self.data_loader.output_train
        if use_data=='all':
            self.input_train=self.data_loader.input
            self.output_train = self.data_loader.output

        self.input_text = self.data_loader.input_text
        self.output_text = self.data_loader.output_text

        self.input = self.data_loader.input
        self.output = self.data_loader.output

        #将数据按照epoch去划分
        self.batch_size=self.config.train_params['batch_size']

        self.input_train=self._split_array_by_epoch_batch(self.input_train,self.batch_size)
        self.output_train = self._split_array_by_epoch_batch(self.output_train, self.batch_size)

    #按照epoch与batch size的规则去划分数据
    def _split_array_by_epoch_batch(self,array, batch_size):
        num_samples = array.shape[0]
        num_batches = num_samples // batch_size

        # 根据 epoch 和 batch size 计算新数组的形状
        new_shape = (batch_size, array.shape[1], num_batches)

        self.data_shape=new_shape

        # 将原始数组调整为新形状
        reshaped_array = array[:num_batches*batch_size,:].reshape(new_shape)

        return reshaped_array




