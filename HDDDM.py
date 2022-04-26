import argparse
from math import sqrt,log
from scipy.stats import t
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from torch.utils.tensorboard import SummaryWriter


class Distance:
    """
    计算Hellinger距离和Jensen-Shannon 散度的
    """
    def hellinger_dist(self,P,Q) ->float:
        """
        计算helling距离
        :param P: 一个窗口中某个特征列对应的概率分布
        :param Q: 下个窗口中每个特征对应的概率分布
        :return: 两者的H距离
        """
        diff=0.
        for key in P.keys():
            diff+=(sqrt(P[key])-sqrt(Q[key]))**2
        return  1/sqrt(2) *sqrt(diff)

def discretizer(data,n_bins,method):
    """
    将数据分bin
    :param data: np.array格式的数据
    :param n_bins: int，确定应该要分多少个bin
    :param method: str ‘equalsize’ 或者 'equalquantile'，前者根据变量的值的跨度进行划分，后者根据区间中值的数量进行划分
    :return: np.array 格式，以n_bins离散化后的数据
    """

    if method=='equalsize':
        return pd.cut(data,n_bins)
    if method=='equalquantile':
        return pd.qcut(data,n_bins)

def process_bin(data,numerical_cols):
    """
    将每一列中的间隔类型转换成整数类型
    :param data: 数据类型为时间间隔和数值型的dataframe
    :param numerical_cols: 其中类型为时间间隔的列名
    :return: 将类型为时间间隔的列转换成整数后返回
    """

    for feature in numerical_cols:
        data[feature]=data[feature].apply(str)
        n=len(data[feature].unique())
        dic={}
        """
        这里直接映射是不是不太好，这样的话不同的批次传过来的数据中 同一个整数可能表示的是不同的间隔？
        """
        for i in range(n):
            dic.update({data[feature].unique()[i]:i})
        data[feature]=data[feature].map(dic)
    return data

def generate_proper_dic(window,union_values):
    """
    本质上是生成概率分布，即 值：概率 的字典
    :param window: 窗口中某个特征所在的列
    :param union_values: 包含两个窗口的该列所有的值的种类
    :return: 返回了每个取值区间对应的概率
    """
    dic={}
    df=window.value_counts()
    n=window.shape[0]

    for key in union_values:
        if key in window.unique():
            dic.update({key:df.loc[key]/n})
        else:
            dic.update({key:0})
    return dic

class HDDDM():
    def __init__(self,data,gamma=1.,alpha=None,distance='Hellinger'):
        """
        :param data: 每次传过来的一批数据
        :param gamma: 第一种方法计算阈值的超参数
        :param alpha: 第二种方法计算阈值的超参数
        :param distance: 距离度量方法
        """
        if gamma is None and alpha is None:
            raise ValueError("gamma 和 alpha不能同时为None，要指定他们其中一个的值")
        self.gamma=gamma
        self.alpha=alpha
        self.n_bins=int(np.floor(np.sqrt(data.shape[0])))
        if distance=='Hellinger':
            # 如果使用H距离的话，就将距离函数设置为h距离函数
            self.distance=Distance().hellinger_dist

        self.baseline=self.add_data(data)
        # 表示当前距离基准数据所在时间步的距离
        self.t_denom=0
        # 这个参数也没什么别的用，是专供beta的第二种求法使用的
        self.n_samples=data.shape[0]
        self.old_dist=0.
        self.epsilons=[]
        # self.epsilons=[0.]
        self.betas=[]

    def add_data(self,data):
        """
        将数据按其所在区间进行重新写入
        :param data: 一个batch的数据
        :return: 返回分好间隔并且转成整数之后的数据
        """
        X=data.copy()
        # 分别分类数据类型和所有数字类别的数据
        X_cat=X.select_dtypes(include='category')
        X_num=X.select_dtypes(include='number')

        data_tmp=pd.DataFrame()
        for c in X_cat.columns:
            data_tmp[c]=X_cat[c]

        for c in X_num.columns:
            data_tmp[c]=discretizer(X_num[c],self.n_bins,'equalsize')
            data_tmp[c]=data_tmp[c].astype('category')

        data_final=process_bin(data_tmp,X_num.columns)
        return data_final

    def windows_distance(self,ref_window,current_window)->float:
        """
        计算两个窗口中数据的
        :param ref_window: 参考窗口的分过区间并且转为整数之后的数据
        :param current_window: 当前窗口的分过区间并且转为整数之后的数据
        :return: 返回两窗口的H距离
        """
        actual_dist=0.
        for feature in self.baseline.columns:
            # 获取当前窗口和参考窗口各列的数据种类，并且将他们整合成一个集合
            ref_liste_values=ref_window[feature].unique()
            current_liste_values=current_window[feature].unique()
            union_values=list(set(ref_liste_values)|set(current_liste_values))

            ref_dic=generate_proper_dic(ref_window[feature],union_values)
            current_dic=generate_proper_dic(current_window[feature],union_values)

            actual_dist+=self.distance(ref_dic,current_dic)
        actual_dist/=len(self.baseline.columns)
        return actual_dist

    def add_new_batch(self,X):
        # 因为本方法求H距离的本质是用特征所在bin在全部数据中出现的次数表示的，所以两者必须数量相同才具有比较性
        # 但是求概率的时候用的是数量/总数 应该不会有影响啊 可能是对beta的第2个求法有影响？
        if int(np.floor(np.sqrt(X.shape[0]))) != self.n_bins:
            raise ValueError('新数据的数据量必须要和基准数据相同')

        self.drift_detected=False
        self.t_denom+=1
        self.curr_batch=self.add_data(X)

        # 计算当前窗口和基准窗口的距离
        curr_dist=self.windows_distance(self.baseline,self.curr_batch)

        n_samples=X.shape[0]

        eps=curr_dist-self.old_dist
        self.epsilons.append(eps)


        # 求epsilon到目前为止的平均值
        if self.t_denom>1:
            """
                    这里也和文章中的不同，文章中是不把当前时间步的数据的epsilon放到其中求平均值的
                    他给出的代码是这样的
                    epsilon_hat = (1. / (self.t_denom)) * np.sum(np.abs(self.epsilons))
                    sigma_hat = np.sqrt(np.sum(np.square(np.abs(self.epsilons) - epsilon_hat)) / (self.t_denom))
            """
            epsilon_hat=(1./(self.t_denom-1))*np.sum(np.abs(self.epsilons[:-1]))
            sigma_hat=np.sqrt(np.sum(np.square(np.abs(self.epsilons[:-1])-epsilon_hat))/(self.t_denom-1))

            # epsilon_hat = (1. / (self.t_denom)) * np.sum(np.abs(self.epsilons))
            # sigma_hat = np.sqrt(np.sum(np.square(np.abs(self.epsilons) - epsilon_hat)) / (self.t_denom))

            beta=0.
            # 下面分别是作者给出的两种求beta的方法
            if self.gamma is not None:
                beta=epsilon_hat+self.gamma*sigma_hat
            else:
                beta=epsilon_hat + t.ppf(1.0 - self.alpha / 2, self.n_samples + n_samples - 2) * sigma_hat / np.sqrt(
                    self.t_denom)

            self.betas.append(beta)

            drift=np.abs(eps)>beta
            if drift==True:
                self.drift_detected=True
                """
                这里作者给的代码是
                self.baseline = self.add_data(X)
                我感觉没必要在做一遍add_data处理，因为上面已经处理过了
                """
                self.baseline=self.curr_batch
                self.t_denom=0
                self.n_samples=X.shape[0]
                self.old_dist=0.
                self.epsilons=[]
                # self.epsilons=[0.]
                self.betas=[]
            else:
                self.n_samples+=n_samples
                self.baseline=pd.concat((self.baseline,self.curr_batch))

if __name__=="__main__":
    argparser=argparse.ArgumentParser('HDDDM')
    argparser.add_argument("--dataset",type=str,default='Datasets/movingRBF.csv')
    args=argparser.parse_args()

    random=1999
    # 将数据分成多少个batch
    n_times=200
    test_fraction=0.25
    gammas=[0.1,0.2,0.3,0.4,0.5]
    alfa=0.1
    errors=[]
    # categorical_variables=['target']

    # 读取数据和数据处理阶段
    Dataset=args.dataset
    Data=pd.read_csv(Dataset)
    Batch=np.array_split(Data,n_times)

    writer = SummaryWriter()
    # 先做一个不用漂移检测的实验看看效果
    model = SGDClassifier()
    model.partial_fit(Batch[0].iloc[:, :-1], Batch[0].iloc[:, -1], classes=np.unique(Batch[0].iloc[:, -1]))
    no_detector_acc = []
    for i in range(1, n_times - 1):
        train_X, train_y = Batch[i].iloc[:, :-1], Batch[i].iloc[:, -1]
        y_pred = model.predict(train_X)
        acc = ((y_pred == train_y).sum()) / len(Batch[i + 1])
        no_detector_acc.append(acc)
        writer.add_scalar("no_detector", acc, i)
        model.partial_fit(train_X, train_y)
    print(f"不用漂移检测平均准确率为：{sum(no_detector_acc) / len(no_detector_acc)}")

    for gamma in gammas:
        drift=[] # 保存检测到的漂移
        accuracy=[] # 保存每个batch上模型的精度
        magnitude=[] # 保存batch之间改变的大小


        # 先用第一批数据训练一下模型
        model = SGDClassifier()
        model.partial_fit(Batch[0].iloc[:, :-1], Batch[0].iloc[:, -1],classes=np.unique(Batch[0].iloc[:, -1]))
        detector=HDDDM(Batch[0].iloc[:, :-1], gamma)

        for i in range(1,n_times-1):
            X_batch = Batch[i].iloc[:, 0:-1]
            y_batch = Batch[i].iloc[:, -1]
            y_pred = model.predict(X_batch)

            acc = ((y_pred == y_batch).sum()) / len(Batch[i+1])
            accuracy.append(acc)
            # 向drift中增加每次的变化距离
            magnitude.append(detector.windows_distance(detector.baseline, detector.add_data(X_batch)))
            # 将新数据放入检测器进行检测，看是否有漂移产生
            detector.add_new_batch(X_batch)

            if detector.drift_detected:
                # print("发生漂移了，重置模型")
                drift.append(i)
                # 需要重置模型
                model=SGDClassifier()
                model=model.partial_fit(X_batch,y_batch,classes=np.unique(y_batch))
            else:
                # print("没有发生漂移")
                model.partial_fit(X_batch,y_batch)

            writer.add_scalars("",{"acc":acc,"magnitude":magnitude[-1],"drift_num":len(drift)},i)
        print(f"gamma:{gamma},acc:{sum(accuracy)/len(accuracy)},drift:{len(drift)}")



