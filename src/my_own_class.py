import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt # 是matplotlib的子包
import os
if os.name == 'posix': # 如果系统是mac或者linux
    plt.rcParams['font.sans-serif'] = ['Songti SC'] #中文字体为宋体
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 在windows系统下显示微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 负号用 ASCII 编码的-显示，而不是unicode的 U+2212
mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)
 
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
import pickle  # 存各种神奇的数据结构
from wordcloud import WordCloud

# import filter
# 
# 从构建解释变量到看结果到划分训练测试到估计
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Lasso                 # 套索回归
from sklearn.model_selection import train_test_split   # 做交叉验证，划分训练集和测试集
from tqdm import trange
from sklearn.preprocessing import StandardScaler # 标准化

# 判断预测准确率的函数
def state_sum(df1,df2):
    df1_state = df1['state'].value_counts().sort_index()
    df2_state = df2['state'].value_counts().sort_index()
    return pd.concat([df1_state,df2_state],axis=1,keys = ['fn1状态计数','fn2状态计数'])

# 整理模型预测的状态分布情况
def pred_state_sum(df1,df2):
    df1_state = df1['FutState'].value_counts().sort_index()#sort_values(ascending=True)
    df2_state = df1['FutPredict'].value_counts().sort_index()
    df3_state = df2['FutState'].value_counts().sort_index()
    df4_state = df2['FutPredict'].value_counts().sort_index()
    return pd.concat([df1_state,df2_state,df3_state,df4_state],axis=1,keys = ['fn1真实状态计数','fn1预测状态计数','fn2真实状态计数','fn2预测状态计数'])

# 打印预测正确率
def accu_rate(df, dataset_str, what_to_predict):
    y_predict = df['FutPredict'].values
    y_real = df['FutState'].values
    print(f"{dataset_str}内{what_to_predict}预测正确率为",f'{(y_real == y_predict).sum()/len(df):.4f}')
    
# 整理模型结果
def PredictSttc(y_df, dataset_str):
    y_df = y_df.astype(float)
    state_num = y_df.max().max()
    print("=" * 50)
    print("最大状态数为",state_num+1)
    print("=" * 50)
    accu_rate(y_df,dataset_str,'状态')

    y_Direction = pd.DataFrame(np.where(y_df > state_num/2, 'Up', np.where(y_df < state_num/2, 'Down', 'Not change')), 
                                index=y_df.index, columns=y_df.columns)
    accu_rate(y_Direction,dataset_str,'涨跌方向')

    y_BuySell = pd.DataFrame(np.where(y_df > state_num/2+1, 'Buy', np.where(y_df < state_num/2-1, 'Sell', 'Not change')), 
                                index=y_df.index, columns=y_df.columns)
    accu_rate(y_BuySell,dataset_str,'买卖方向')
    
    y_real_BuySell = y_BuySell.copy()
    y_real_BuySell = y_real_BuySell[y_real_BuySell != 'Not change'].dropna()
    y_real_BuySell = y_real_BuySell.groupby(y_real_BuySell.index.date).apply(lambda x: x[x['FutPredict'] != x['FutPredict'].shift(1)])
    print("模型每日去重开仓信号次数",f"{len(y_real_BuySell)/len(set(y_df.index.date)):.4f}")
    print("=" * 50)

# 存模型
def pickle_save(model2save,model_name=""):
    with open(f'./model{model_name}.pickle', 'wb') as f:  # 存结果
        pickle.dump(model2save, f)
    with open(f'./model{model_name}.pickle', 'rb') as f:  # 读结果
        factor_rslt = pickle.load(f)
    print(type(factor_rslt))
    return(factor_rslt)

# Dummy Predict概率
def dummy_predict_sttc(df,df_name):
    print('='*30)
    print(f"{df_name}上涨概率为{(df>0).sum()/len(df):.4f}")
    print(f"{df_name}不变概率为{(df==0).sum()/len(df):.4f}")
    print(f"{df_name}下跌概率为{(df<0).sum()/len(df):.4f}")
    print('='*30)
    
# 读数据
def read_data(filename):
    fn1=pd.read_csv(f'./dataset/{filename}_1.csv',parse_dates=[2])
    fn2=pd.read_csv(f'./dataset/{filename}_2.csv',parse_dates=[2])
    fn3=pd.read_csv(f'./dataset/{filename}_3.csv',parse_dates=[2])
    fn4=pd.read_csv(f'./dataset/{filename}_4.csv',parse_dates=[2])
    fn = pd.concat([fn1,fn2,fn3,fn4])
    fn = fn.set_index('Time')
    fn.index = pd.to_datetime(fn.index)
    fn = fn[~fn.index.duplicated(keep='first')]
    return fn

def my_plot(x,y,plot_index,title,xlabel,ylabel,xscale,save_name):
    fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(10, 5) )
    fontsize = 16
    plot_index = np.logspace(-4, -2, num=13)

    axes.set_title(f'{title}',fontsize = fontsize)
    axes.plot(x,y,'.')
    axes.set_xlabel(f'{xlabel}', fontsize = fontsize)
    axes.set_ylabel(f'{ylabel}', fontsize = fontsize)
    axes.set_xscale(f'{xscale}')
    # axes.set_yscale('log')
    axes.grid()
    axes.set_xticks(plot_index)
    axes.set_xticklabels([str(f"{x:.2e}") for x in plot_index],rotation = 30)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    plt.savefig(f'./result/{save_name}_plot.jpg', bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    plt.show()

# 词云
def word_plot_hah(df,plot_name):
    df.columns = ['words','sizes']
    word_sizes = dict(zip(df['words'], df['sizes']))
    wc = WordCloud(width=800, height=400, background_color='white', max_words=50,random_state=230412,
                relative_scaling=0.5, prefer_horizontal=0.8, colormap='viridis').generate_from_frequencies(word_sizes)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud of {plot_name}", fontsize=30)
    plt.tight_layout(pad=0)
    plt.savefig(f'./result/{plot_name}_word_plot.jpg', bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    plt.show()


# # TODO resample
# def data_resample(fn):
#     frequency = '1min'
#     fn_resample = fn.resample(frequency).last()
#     fn_resample['Open'] = fn['Open'].resample(frequency).first()
#     fn_resample['High'] = fn['High'].resample(frequency).max()
#     fn_resample['Low'] = fn['Low'].resample(frequency).min()
#     return fn_resample

# # TODO filter
# def data_filter(fn):
#     for clmn in fn.columns:
#         fn.loc[:,clmn] = filter.kalmanfilter(fn1.loc[:,clmn],u,v)

# lasso
class my_lasso_class():
    def __init__(self,df1,df2,features,label,save_name): # df1是训练和测试集，按0.8分，df2是测试集
        self.features = features
        self.label = label
        self.df1 = df1
        self.df2 = df2
        self.save_name = save_name
        
        print('='*30)
        print(f'result for {save_name}')
        self.lasso_data_process()
        self.l1_penalty_search()
        self.l1_penalty_plot()
        
    def lasso_data_process(self): 
        train_validate_seperate = self.df1.index[int(len(self.df1) * 0.8)] # 区分训练集和测试集
        
        # 划分训练集、验证集、测试集
        self.X_lasso_train = self.df1.loc[:train_validate_seperate,self.features]
        self.y_lasso_train = self.df1.loc[:train_validate_seperate,self.label]
        self.X_lasso_validate = self.df1.loc[train_validate_seperate:,self.features]
        self.y_lasso_validate = self.df1.loc[train_validate_seperate:,self.label]
        self.X_lasso_test = self.df2.loc[:,self.features] 
        self.y_lasso_test = self.df2.loc[:,self.label]

        # 标准化
        scaler = StandardScaler()
        df_scaled = scaler.fit(self.X_lasso_train)
        self.X_lasso_train = pd.DataFrame(df_scaled.transform(self.X_lasso_train),index = self.df1.index[:int(len(self.df1) * 0.8)+1])
        self.X_lasso_train.columns = self.features
        self.X_lasso_validate = pd.DataFrame(df_scaled.transform(self.X_lasso_validate),index = self.X_lasso_validate.index)
        self.X_lasso_validate.columns = self.features
        self.X_lasso_test = pd.DataFrame(df_scaled.transform(self.X_lasso_test),index = self.df2.index)
        self.X_lasso_test.columns = self.features
        
        print( '训练集的特征大小为', self.X_lasso_train.shape )
        print( '训练集的标签大小为', self.y_lasso_train.shape )
        print( '验证集的特征大小为', self.X_lasso_validate.shape )
        print( '验证集的标签大小为', self.y_lasso_validate.shape )
        print( '测试集的特征大小为', self.X_lasso_test.shape )
        print( '测试集的标签大小为', self.y_lasso_test.shape )

    def l1_penalty_search(self):
        len_lst = list()
        valid_error = list()
        l1_penalty_set = np.logspace(-4, -2, num=13) # 简单记录一下：(shift(-1),13,18, 0.03729, 0.0343), (shift(-2),13,24,0.05857, 0.05462)
        for i in range(len(l1_penalty_set)):
            l1_penalty = l1_penalty_set[i]
            model = self.lasso_regression( self.X_lasso_train, self.y_lasso_train, l1_penalty )
            y_real = self.y_lasso_validate.astype(float).values
            y_predict = model.predict( self.X_lasso_validate.astype(float).values )
            i_valid_error = ((y_real- y_predict)**2).sum()/len(y_predict)

            valid_error.append( i_valid_error )
            var_lst = self.print_coefficients( model, self.features )
            var_lst[var_lst != 0].dropna()['name'].values[1:]
            len_lst.append(len(var_lst[var_lst != 0].dropna()['name'].values[1:]))
            # print( "lambda: %.5f, validation error: %.10e" %(l1_penalty, i_valid_error) )

        best_l1_penalty = l1_penalty_set[ np.argmin(valid_error) ]
        model = self.lasso_regression(self.X_lasso_train, self.y_lasso_train, best_l1_penalty )

        best_valid_error = min(valid_error)
        test_error = ((self.y_lasso_test.values - model.predict( self.X_lasso_test.values ))**2).sum() /len(self.X_lasso_test)
        num_of_features = len_lst[np.argmin(valid_error)]
        self.valid_error = valid_error
        self.l1_penalty_set = l1_penalty_set
        
        print(f"best validation error: {best_valid_error:.5f} ")
        print(f'corresponding test error:{test_error:.5f}')
        print(f"best lambda: {best_l1_penalty:.5f}")
        print(f"number of the features: {int(num_of_features)}")

        var_lst = self.print_coefficients( model, self.features )
        self.X_list = var_lst[var_lst != 0].dropna()['name'].values[1:]
        print(self.X_list)

    def l1_penalty_plot(self):
        my_plot(self.l1_penalty_set, self.valid_error,np.logspace(-4, -2, num=13),'L1惩罚参数选择','参数范围','验证集均方误差','log',self.save_name)
        
    def lasso_regression(self, X_train, y_train, l1_penalty ):
        model = Lasso( alpha=l1_penalty)
        model.fit( X_train.values, y_train.values )
        return model

    def print_coefficients(self, model, features ):        
        w = list( np.hstack((model.intercept_,model.coef_)) )
        labels = ['intercept'] + features
        df = pd.DataFrame({'name': labels, 'value': w})
        return df

# probit
class my_probit_class():
    def __init__(self,df1,df2,features,label,save_name): # df1是训练和测试集，按0.8分，df2是测试集
        self.features = features
        self.label = label
        self.df1 = df1
        self.df2 = df2
        self.save_name = save_name
        
        print('='*30)
        print(f'result for {save_name}')
        self.probit_data_process()
        self.probit_model() # 四分钟，还挺久的
        self.probit_df_cal()
        
    def probit_data_process(self): 
        # 划分训练集、验证集、测试集
        self.X_probit_train = self.df1.loc[:,self.features]
        self.y_probit_train = self.df1.loc[:,self.label]
        self.X_probit_test = self.df2.loc[:,self.features] # fn2整体作为验证集
        self.y_probit_test = self.df2.loc[:,self.label]

        # 标准化
        scaler = StandardScaler()
        df_scaled = scaler.fit(self.X_probit_train)
        self.X_probit_train = pd.DataFrame(df_scaled.transform(self.X_probit_train),index = self.df1.index)
        self.X_probit_train.columns = self.features
        self.X_probit_test = pd.DataFrame(df_scaled.transform(self.X_probit_test),index = self.df2.index)
        self.X_probit_test.columns = self.features
        
        print( '训练集的特征大小为', self.X_probit_train.shape )
        print( '训练集的标签大小为', self.y_probit_train.shape )
        print( '测试集的特征大小为', self.X_probit_test.shape )
        print( '测试集的标签大小为', self.y_probit_test.shape )
        
    def probit_model(self):
        self.mod_prob = OrderedModel(self.y_probit_train, self.X_probit_train, distr='probit').fit(method='bfgs')
        self.probit_t_values = pd.DataFrame(self.mod_prob.tvalues[self.features]).reset_index()
        self.probit_t_values.loc[:,0] = abs(self.probit_t_values.loc[:,0])
        self.probit_t_values.sort_values(by = 0,ascending = False,inplace = True)
        
    def predict_sttc(self,prediction_X,prediction_y):
        predicted = self.mod_prob.model.predict(self.mod_prob.params, exog=np.array(prediction_X))
        y_predict_probit_train = pd.DataFrame(predicted,index = prediction_y.index).idxmax(axis = 1)

        prediction_df = pd.DataFrame(index = prediction_y.index)
        
        prediction_df['FutState'] = prediction_y
        prediction_df['FutPredict'] = y_predict_probit_train

        # prediction_df.to_csv(f'./result/{self.save_name}_{pred_name}.csv')
        return prediction_df

    def probit_df_cal(self):
        self.train_df = self.predict_sttc(self.X_probit_train,self.y_probit_train,)
        self.test_df = self.predict_sttc(self.X_probit_test,self.y_probit_test)
        self.state_sum_df = pred_state_sum(self.train_df,self.test_df)

    def train_predict_sttc(self):
        PredictSttc(self.train_df,'训练集')
        PredictSttc(self.test_df,'测试集')

    def probit_word_plot(self):
        word_plot_hah(self.probit_t_values,self.save_name)

# adaboost
class my_adaboost_class():
    def __init__(self,df1,df2,features,label,save_name): # df1是训练集，df2是测试集
        self.features = features
        self.label = label
        self.df1 = df1
        self.df2 = df2
        self.save_name = save_name
        
        print('='*30)
        print(f'result for {save_name}')
        self.adaboost_data_process()
        self.adaboost_model() # 
        self.adaboost_df_cal()
        
    def adaboost_data_process(self): 
        # 划分训练集、验证集、测试集
        self.X_adaboost_train = self.df1.loc[:,self.features]
        self.y_adaboost_train = self.df1.loc[:,self.label]
        self.X_adaboost_test = self.df2.loc[:,self.features] # fn2整体作为验证集
        self.y_adaboost_test = self.df2.loc[:,self.label]

        # 标准化
        scaler = StandardScaler()
        df_scaled = scaler.fit(self.X_adaboost_train)
        self.X_adaboost_train = pd.DataFrame(df_scaled.transform(self.X_adaboost_train),index = self.df1.index)
        self.X_adaboost_train.columns = self.features
        self.X_adaboost_test = pd.DataFrame(df_scaled.transform(self.X_adaboost_test),index = self.df2.index)
        self.X_adaboost_test.columns = self.features
        
        print( '训练集的特征大小为', self.X_adaboost_train.shape )
        print( '训练集的标签大小为', self.y_adaboost_train.shape )
        print( '测试集的特征大小为', self.X_adaboost_test.shape )
        print( '测试集的标签大小为', self.y_adaboost_test.shape )
        
    def adaboost_model(self):
        DT = DecisionTreeClassifier( max_depth=3, random_state=230408 )
        Ensemble = AdaBoostClassifier( DT, n_estimators=100, random_state=230408 )
        Ensemble.fit(self.X_adaboost_train, self.y_adaboost_train)
        self.model_save = Ensemble
        # print( Ensemble.n_estimators )
        # Ensemble.estimators_
        self.y_predict_adaboost_train, self.y_predict_adaboost_test = Ensemble.predict(self.X_adaboost_train), Ensemble.predict(self.X_adaboost_test)
        self.adaboost_feature_compare = pd.DataFrame({'变量名':self.features,'权重':self.model_save.feature_importances_}).sort_values('权重',ascending=False).T   
        
    def predict_sttc(self,y_real,y_predict):
        prediction_df = pd.DataFrame(index = y_real.index)
        prediction_df['FutState'] = y_real
        prediction_df['FutPredict'] = y_predict
        return prediction_df

    def adaboost_df_cal(self):
        self.train_df = self.predict_sttc(self.y_adaboost_train,self.y_predict_adaboost_train,)
        self.test_df = self.predict_sttc(self.y_adaboost_test,self.y_predict_adaboost_test)
        self.state_sum_df = pred_state_sum(self.train_df,self.test_df)

    def train_predict_sttc(self):
        PredictSttc(self.train_df,'训练集')
        PredictSttc(self.test_df,'测试集')
    
    def adaboost_word_plot(self):
        word_plot_hah(self.adaboost_feature_compare.T,self.save_name)
    
if __name__ == '__main__':
    # 读取原数据，fn是future index
    fn1 = read_data(201405)
    fn2 = read_data(201406)
    print(len(fn1), len(fn2))
    np.where(fn1.index.duplicated()) # 看看有没重复数据
    
    fn1 = fn1.resample('d').last().dropna()
    fn2 = fn2.resample('d').last().dropna()

    all_features = ['Price','TotalAmount','BV5']
    label = 'BV1'
    # ada_shift2 = my_adaboost_class(fn1,fn2,all_features,label,'BV1')
    # ada_shift2.state_sum_df#.sort_index(ascending=True)
    # ada_shift2.adaboost_feature_compare#.T.sort_values('权重',ascending=False).T  
    # ada_shift2.lasso_word_plot()

    prob_shift2 = my_probit_class(fn1,fn2,all_features,label,'BV1')
    prob_shift2.state_sum_df#.sort_index(ascending=True)
    len(prob_shift2.state_sum_df)
    prob_shift2.probit_t_values#.info()
    prob_shift2.probit_word_plot()