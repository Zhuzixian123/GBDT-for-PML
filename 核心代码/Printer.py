import time

import tqdm

from .Utils import add_spaces


class init_printer:
    def __init__(self,dict_list):
        """
        格式预览：
        +===============================--- 数 据 处 理 ---===============================+
        | ============================================================================= |
        |     [progress]     [step]        [progress rate]                              |
        |        读取数据       5/5    : 100%|██████████| 5/5 [00:05<00:00,  1.01s/it]    |
        |        检测数据       5/5    : 100%|██████████| 5/5 [00:05<00:00,  1.01s/it]    |
        |        处理数据       5/5    : 100%|██████████| 5/5 [00:05<00:00,  1.01s/it]    |
        |     导出训练数据       5/5    : 100%|██████████| 5/5 [00:05<00:00,  1.01s/it]    |
        |===============================================================================|
        :param dict_list:
        """

        # 使用列表推导式提取字典中的属性，并组成一个字符数组
        name_array = [d['名称（name）'] for d in dict_list]
        step_array = [d['步骤（step）'] for d in dict_list]

        #创建字典用于对应进度条与name
        self.name_bars={}

        #创建格式化表格
        print('+'+'=' * 31+'--- 数 据 处 理 ---'+'=' * 31+'+')
        print('|','='*77,'|')
        print('|',"{:>{position}}".format('[progress]', position=14),
                  "{:>{position}}".format('[step]', position=10),
                  "{:>{position}}".format('[progress rate]', position=22),
                  "{:>{position}}".format('|', position=30))

        time.sleep(0.1)
        for name,step in zip(name_array,step_array):
            self.name_bars[name] = self.aotu_bar(name,step)

    def all_last(self):
        print('|'+ '=' * 79+'|')
        print(' ')

    #生成一批进度条
    class aotu_bar:
        def __init__(self,name,all_step):

            self.name=name
            self.now_step=0
            self.all_step = all_step
            self.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"+' '*4+'|'


        def open_it(self):
            self.step_bar = f"{self.now_step}/{self.all_step}"+' '*4
            self.bar = tqdm.tqdm(total=self.all_step,bar_format=self.bar_format,desc='|'+
                                "{:>{position}}".format(self.name, position=14-int(len(self.name)/2))+
                                "{:>{position}}".format(self.step_bar, position=14))

        def update_it(self):
            self.now_step+=1
            self.step_bar=f"{self.now_step}/{self.all_step}"+' '*4
            self.bar.set_description('|'+"{:>{position}}".format(self.name, position=14-int(len(self.name)/2))+
                                         "{:>{position}}".format(self.step_bar, position=14))
            self.bar.update(1)

        def close_it(self):
            self.bar.close()

class run_printer:
    def __init__(self,data_shape):
        #创建格式化表格
        print('+'+'=' * 31+'--- 模 型 训 练 ---'+'=' * 31+'+')
        print('|','='*77,'|')
        print('|',"{:>{position}}".format('[progress]', position=14),
                  "{:>{position}}".format('[step]', position=10),
                  "{:>{position}}".format('[progress rate]', position=22),
                  "{:>{position}}".format('|', position=30))

        self.epoch=0
        self.num_batch=data_shape[2]
        self.batch_size=data_shape[0]
        self.all_epoch=100

        self.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]" + ' ' * 4 + '|'



    def in_train(self):
        self.now_batch, self.now_step = 0, 0

        self.bar=tqdm.tqdm(total=self.batch_size*self.num_batch,
                           mininterval=0.5,bar_format=self.bar_format,
                           leave=False,
                           desc='|'+
                                "{:>{position}}".format(self.epoch,position=10) + '/' + str(self.all_epoch) +
                                "{:>{position}}".format(self.now_batch, position=10)+'/'+str(self.num_batch)+
                                "{:>{position}}".format(str(0), position=10))


        self.epoch+=1#进入下一个epoch
        self.batch=0

        pass

    def step_train(self,loss):
        self.batch+=1
        self.loss=loss

        #记录当前批次，以及当前步骤
        self.now_batch,self.now_step=self.batch//self.batch_size,self.batch%self.batch_size

        self.bar.set_description('|'+
                                 "{:>{position}}".format(self.epoch, position=5) + '/' + str(self.all_epoch) +
                                 "{:>{position}}".format(self.now_batch, position=6)+'/'+str(self.num_batch)+
                                 "{:>{position}}".format(str(loss)[:5], position=6))
        self.bar.update(1)

        pass

    def out_train(self):
        self.bar.close()
        pass

    def print_result(self,loss,score):
        print(f"|   {self.epoch}/{100}   正确率：{score}   损失：{loss}  |"  )

    def in_test(self):
        pass

    def step_test(self):
        pass

    def out_test(self):
        pass