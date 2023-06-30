import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from .Printer import init_printer

class data_loader:
    def __init__(self,config):

        #参数配置
        self.conifg=config

        #初始化模块,开启打印机
        read_data=self._read_data(self,self.conifg)
        detect_data=self._detect_data(self,self.conifg)
        initical_setdata=self._initical_setdata(self,self.conifg)
        output_data=self._output_data(self,self.conifg)

        self.init_printer=init_printer([
            read_data.info,
            detect_data.info,
            initical_setdata.info,
            output_data.info
        ])

        #读取数据
        read_data.run(self.init_printer)
        #数据检测
        detect_data.run(self.init_printer)
        #数据预处理
        initical_setdata.run(self.init_printer)
        #数据最后导出
        output_data.run(self.init_printer)

        #进度条显示结尾
        self.init_printer.all_last()


    class _read_data:
        def __init__(self,up_self,config):
            self.info={
                '名称（name）':'读取数据',
                '步骤（step）':4,
            }
            self.config=config
            self.up_self=up_self
            self.data_path = self.config.base_params['data_path']

        def run(self,init_printer):
            init_printer.name_bars[self.info['名称（name）']].open_it()

            c = self.config.base_params['ceng']
            if os.path.exists(self.data_path):
                if self.config.base_params['time_data']:
                    input_ex = pd.read_csv(self.data_path+'input_data_ex.csv', header=None,
                                           index_col=None).values[:, 960 * (c - 1):960 * c].reshape(240 * 1230, 4)
                    timedata = np.repeat(np.arange(1, 1231), 240).reshape(-1, 1)
                    input_ex = np.concatenate((input_ex, timedata), axis=1)
                    init_printer.name_bars[self.info['名称（name）']].update_it()
                    input_ey = pd.read_csv(self.data_path+'input_data_ey.csv', header=None,
                                           index_col=None).values[:, 960 * (c - 1):960 * c].reshape(240 * 1230, 4)
                    init_printer.name_bars[self.info['名称（name）']].update_it()
                    input_hz = pd.read_csv(self.data_path+'input_data_hz.csv', header=None,
                                           index_col=None).values[:, 960 * (c - 1):960 * c].reshape(240 * 1230, 4)
                    init_printer.name_bars[self.info['名称（name）']].update_it()
                    output_ex = pd.read_csv(self.data_path+'output_data_hz.csv', header=None,
                                            index_col=None).values[:, 240 * (c - 1):240 * c].reshape(240 * 1230, 1)
                    init_printer.name_bars[self.info['名称（name）']].update_it()
                else:
                    input_ex = pd.read_csv(self.data_path+'input_data_ex.csv', header=None,
                                           index_col=None).values[:, 960 * (c - 1):960 * c].reshape(240 * 1230, 4)
                    init_printer.name_bars[self.info['名称（name）']].update_it()
                    input_ey = pd.read_csv(self.data_path+'input_data_ey.csv', header=None,
                                           index_col=None).values[:, 960 * (c - 1):960 * c].reshape(240 * 1230, 4)
                    init_printer.name_bars[self.info['名称（name）']].update_it()
                    input_hz = pd.read_csv(self.data_path+'input_data_hz.csv', header=None,
                                           index_col=None).values[:, 960 * (c - 1):960 * c].reshape(240 * 1230, 4)
                    init_printer.name_bars[self.info['名称（name）']].update_it()

                    output_ex = pd.read_csv(self.data_path+'output_data_hz.csv', header=None,
                                            index_col=None).values[:, 240 * (c - 1):240 * c].reshape(240 * 1230, 1)
                    init_printer.name_bars[self.info['名称（name）']].update_it()

                self.up_self.input = np.concatenate((input_ex, input_ey, input_hz), axis=1)
                self.up_self.output = output_ex
                self.up_self.outr = self.up_self.output.reshape(-1)
            else:
                print(self.data_path)
                warnings.warn("=====-- 没有数据文件，请使用data_reset生成数据 --=====")
                sys.exit(1)

            init_printer.name_bars[self.info['名称（name）']].close_it()
            pass

    class _detect_data:
        def __init__(self,up_self,config):
            self.info = {
                '名称（name）': '检测数据',
                '步骤（step）': 5,
            }


        def run(self,init_printer):
            init_printer.name_bars[self.info['名称（name）']].open_it()
            for i in range(5):
                time.sleep(1)
                init_printer.name_bars[self.info['名称（name）']].update_it()
            init_printer.name_bars[self.info['名称（name）']].close_it()
            pass

    class _initical_setdata:
        def __init__(self,up_self,config):
            self.info = {
                '名称（name）': '处理数据',
                '步骤（step）': 4,
            }
            self.config=config
            self.up_self=up_self


        def run(self,init_printer):
            init_printer.name_bars[self.info['名称（name）']].open_it()

            if self.config.train_params['train_model']:
                self.up_self.means_input = np.mean(self.up_self.input, axis=0)
                self.up_self.stds_input = np.std(self.up_self.input, axis=0)
                self.up_self.input = (self.up_self.input - self.up_self.means_input) / self.up_self.stds_input
                init_printer.name_bars[self.info['名称（name）']].update_it()

                self.up_self.means_output = np.mean(self.up_self.output, axis=0)
                self.up_self.stds_output = np.std(self.up_self.output, axis=0)
                self.up_self.output = (self.up_self.output - self.up_self.means_output) / self.up_self.stds_output
                init_printer.name_bars[self.info['名称（name）']].update_it()

                df_in = pd.DataFrame(self.up_self.means_output)
                df_in.to_csv('./Result/out_means.csv', index=False, header=False)
                df_in = pd.DataFrame(self.up_self.stds_output)
                df_in.to_csv("./Result/out_stds.csv", index=False, header=False)
                init_printer.name_bars[self.info['名称（name）']].update_it()

                df_in = pd.DataFrame(self.up_self.means_input)
                df_in.to_csv('./Result/in_means.csv', index=False, header=False)
                df_in = pd.DataFrame(self.up_self.stds_input)
                df_in.to_csv("./Result/in_stds.csv", index=False, header=False)
                init_printer.name_bars[self.info['名称（name）']].update_it()
            else:
                self.up_self.means_input = pd.read_csv('.\Result\in_means.csv',
                                               header=None, index_col=None).values
                init_printer.name_bars[self.info['名称（name）']].update_it()
                self.up_self.stds_input = pd.read_csv('.\Result\in_stds.csv',
                                              header=None, index_col=None).values
                init_printer.name_bars[self.info['名称（name）']].update_it()
                self.up_self.means_output = pd.read_csv('.\Result\out_means.csv',
                                                header=None, index_col=None).values
                init_printer.name_bars[self.info['名称（name）']].update_it()
                self.up_self.stds_output = pd.read_csv('.\Result\out_stds.csv',
                                               header=None, index_col=None).values
                init_printer.name_bars[self.info['名称（name）']].update_it()

                self.up_self.input = (self.up_self.input - self.up_self.means_input.reshape(-1)) / self.up_self.stds_input.reshape(-1)
                self.up_self.output = (self.up_self.output - self.up_self.means_output.reshape(-1)) / self.up_self.stds_output.reshape(-1)
            init_printer.name_bars[self.info['名称（name）']].close_it()
            pass

    class _output_data:
        def __init__(self,up_self,config):
            self.info = {
                '名称（name）': '导出训练数据',
                '步骤（step）': 1,
            }
            self.config=config
            self.up_self=up_self


        def run(self,init_printer):
            init_printer.name_bars[self.info['名称（name）']].open_it()

            self.up_self.text_size = self.config.base_params['text_size']  # 测试集比例
            self.up_self.random_seed = self.config.base_params['random_seed']  # 随机种子
            self.up_self.input_train, self.up_self.input_text, self.up_self.output_train, self.up_self.output_text = \
                train_test_split(self.up_self.input, self.up_self.output, test_size=self.up_self.text_size, random_state=self.up_self.random_seed)

            init_printer.name_bars[self.info['名称（name）']].update_it()

            init_printer.name_bars[self.info['名称（name）']].close_it()
            pass