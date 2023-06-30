# 使用文档

***

此文档将介绍和提供**训练/调试**的完整方案。

本项目使用完整成熟的SK-Learn库构建基于GBDT的预测模型。

浏览下面的快速开始。

## **快速开始**

***

### 安装

***

克隆项目并且在Python>=3.7.0的环境下安装依赖requirements.txt。

```python
pip install -r requirements.txt  # install
```

### 数据集配置

***

1、从百度网盘获取生成后的数据集文件

```python
链接：https://pan.baidu.com/s/1iXQV9VTNz0VoYB-PYvEHEg?pwd=1111 
提取码：1111
```

2、在配置文件config.py中设置数据处理方式，包括数据地址

```python
self.base_params={
            'normalize_data':True,            #数据归一化
            'log10_data': False,              #数据取对数
            'time_data': True,                #增加时间数据
            'text_size':0.2,                  #数据集比例
            'random_seed':21,                 #数据随机种子
            'ceng': 1,                        #数据使用的3D层数
            'data_path':'E:/pycharm_site/代码项目/课题1/数据文件/', #数据地址
            }
```

### 运行

***

*   直接使用运行main.py即可。

    ```python
    main.py
    ```

### 结果

***

在配置文件config.py中设置模型更多内容。

1、模型测试结果保存，将保存以四个超参数命名的数据结果，内容为每个迭代过程的正确率

```python
self.train_params={
            'save_result':True,
            }
```

2、模型保存

```python
self.train_params={
            'model_name': 'base_line',
            'save_model':True,
            }
```

3、模型加载

```python
self.train_params={
            'load_model':True,
            'model_path':'last_model.pkl',
            'load_model_name':'base_line',
            }
```

## **联系我们**

***

对于本项目的错误报告及功能请求请邮箱联系我们：<508834637@qq.com>
