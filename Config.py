class config:
    def __init__(self):

        #GBDT模型超参数
        self.config1={
            'n_estimators': 60,         # 弱学习器（决策树）的数量
            'learning_rate': 0.2,       # 学习率
            'max_depth': 10,            # 决策树的最大深度
            'min_samples_split': 8,     # 内部节点分裂所需的最小样本数
            'subsample': 0.9,           # 子样本的抽样比例
            'loss': 'squared_error',    # 损失函数（平方损失）
            # 'loss':custom_loss,
            'random_state': 42          # 随机种子（用于重现结果）
        }

        self.base_params={
            'device':'cuda',
            'normalize_data':True,
            'log10_data': False,
            'time_data': True,
            'text_size':0.2,
            'random_seed':21,

            'ceng': 1,

            'data_path':'E:/pycharm_site/代码项目/课题1/数据文件/',
            }

        self.model_params={
            'data_type': 'float64',
            'num_layers':4,#3 or 4
            'first_output':64,
            'hidden_size':32,
            'last_output':64,
            }

        self.train_params={
            'learning_rate':0.03,
            'scheduler':True,
            'max_step':3,

            'run_num':100,
            'epoch_max':100,
            'batch_size':40000,
            'train_model':True,
            'use_data':'train',

            'model_name': 'base_line',
            'model_id': 'new-4',

            'load_model':False,
            'model_path':'E:\\pycharm_site\\代码项目\\课题1_新架构\\模型集合\\传统算法\\方案一-单点\\训练结果\\base_line\\last_model.pkl',
            'load_model_name':'base_line',
            'load_model_id':'1',

            'save_model':True,
            'save_result':True,
            }