import pandas as pd
import numpy as np
from abc import abstractmethod, ABCMeta
from collections import abc
import pickle
from itertools import zip_longest
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import sys

import tqdm

import itertools
import multiprocessing
import dill
from functools import wraps

class SiMuPaiPaiWang():
    """
    排排网的配色
    """
    colors = {'strategy': '#de3633',
              'benchmark': '#80b3f6',
              'excess': '#f4b63f'}

    def __getitem__(self, key):
        return self.colors[key]

    def __repr__(self):
        return self.colors.__repr__()

# TODO: 增加stop_loss


class DataSet(object):
    def __init__(self):
        pass


class Displayer(object):
    """
    Display a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        assert isinstance(df.index, pd.DatetimeIndex)
        assert 'benchmark_curve' in df.columns
        assert 'strategy_curve' in df.columns
        assert 'position' in df.columns
        assert 'signal' in df.columns
        self.df = df
        self.freq_base = self.df['benchmark'].resample('d').apply(lambda x: len(x))[0]
        self.holding_infos = self._calc_holding_infos()
        self.out_stats = self._calc_stats()

    def _calc_holding_infos(self):
        state = self.df['position'].copy(deep=True)
        assert not all(state == 0), "没有进行交易, 故不进行统计"

        time_info = [num for num, count in enumerate(np.abs((state - state.shift(1)).fillna(0)))
                     for i in range(int(count))]
        open_time = time_info[::2]
        exit_time = time_info[1::2]
        holding_infos = pd.DataFrame(zip_longest(open_time, exit_time, fillvalue=None))
        holding_infos.columns = ['open_time', 'exit_time']
        holding_infos.fillna(len(self.df), inplace=True)
        holding_infos['direction'] = state[list(holding_infos['open_time'])].values
        holding_infos['holding_time'] = holding_infos['exit_time'] - holding_infos['open_time']
        holding_infos['returns'] = holding_infos.apply(lambda x:
                                                       np.log(self.df['strategy_curve'].iloc[int(x['exit_time']) - 1] /
                                                              self.df['strategy_curve'].iloc[int(x['open_time']) - 1]),
                                                       axis=1)

        return holding_infos

    def _calc_stats(self):
        output_stat = {}
        strategy_returns = np.log(self.df['strategy_curve'] / self.df['strategy_curve'].shift(1))
        benchmark_returns = np.log(self.df['benchmark_curve'] / self.df['benchmark_curve'].shift(1))
        excess_returns = strategy_returns - benchmark_returns
        output_stat['Annualized_Mean'] = 252 * self.freq_base * np.mean(strategy_returns)
        output_stat['Annualized_Std'] = np.sqrt(252 * self.freq_base) * np.std(strategy_returns)
        output_stat['Sharpe'] = output_stat['Annualized_Mean'] / output_stat['Annualized_Std']
        output_stat['Excess_Annualized_Mean'] = 252 * self.freq_base * np.mean(excess_returns)
        output_stat['Excess_Annualized_Std'] = np.sqrt(252 * self.freq_base) * np.std(excess_returns)
        output_stat['Excess_sharpe'] = output_stat['Excess_Annualized_Mean'] / output_stat['Excess_Annualized_Std']
        output_stat['MaxDrawDown'] = ((self.df['strategy_curve'].cummax() - self.df['strategy_curve']) / self.df[
            'strategy_curve'].cummax()).max()
        try:
            output_stat['LongCounts'] = self.holding_infos['direction'].value_counts()[1]
            output_stat['MeanLongTime'] = \
            self.holding_infos['holding_time'].groupby(self.holding_infos['direction']).mean()[1]
            output_stat['PerLongReturn'] = \
            self.holding_infos['returns'].groupby(self.holding_infos['direction']).mean()[1]

        except KeyError:
            output_stat['LongCounts'] = 0
            output_stat['MeanLongTime'] = 0
            output_stat['PerLongReturn'] = 0

        try:
            output_stat['ShortCounts'] = self.holding_infos['direction'].value_counts()[-1]
            output_stat['MeanShortTime'] = \
            self.holding_infos['holding_time'].groupby(self.holding_infos['direction']).mean()[-1]
            output_stat['PerShortReturn'] = \
            self.holding_infos['returns'].groupby(self.holding_infos['direction']).mean()[-1]
        except KeyError:
            output_stat['ShortCounts'] = 0
            output_stat['MeanShortTime'] = 0
            output_stat['PerShortReturn'] = 0

        try:
            temp_p = self.holding_infos['returns'][self.holding_infos['returns'] > 0].mean()
            temp_n = self.holding_infos['returns'][self.holding_infos['returns'] < 0].mean()
            output_stat['PnL'] = np.abs(temp_p / temp_n)

        except ZeroDivisionError:
            output_stat['PnL'] = np.inf

        output_stat['WinRate'] = (self.holding_infos['returns'] > 0).sum() / len(self.holding_infos['returns'])

        return pd.Series(output_stat)

    def plot_(self, tick_count=12):
        datetime_index = self.df.index
        strategy_returns = self.df['strategy_curve'].pct_change()
        benchmark_returns = self.df['benchmark_curve'].pct_change()
        excess_returns = strategy_returns - benchmark_returns
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = gridspec.GridSpec(4, 2, left=0.04, bottom=0.15, right=0.96, top=0.96, wspace=None, hspace=0,
                               height_ratios=[5, 1, 1, 1])
        ax = fig.add_subplot(gs[0, :])
        position = fig.add_subplot(gs[1, :])
        signal = fig.add_subplot(gs[2, :])
        ax3 = fig.add_subplot(gs[3, :])
        # table_stat = fig.add_subplot(gs[2:, 0])
        ax.plot(range(len(self.df)), strategy_returns.cumsum().fillna(0), color=SiMuPaiPaiWang()['strategy'])
        ax.plot(range(len(self.df)), benchmark_returns.cumsum().fillna(0), color=SiMuPaiPaiWang()['benchmark'])
        ax.plot(range(len(self.df)), excess_returns.cumsum().fillna(0), color=SiMuPaiPaiWang()['excess'])
        ax.hlines(y=0, xmin=0, xmax=len(self.df), color='grey', linestyles='dashed')
        step = int(len(self.df) / tick_count)
        ax.set_xlim(0, len(self.df))
        ax.set_xticks([])
        max_drawdowns = (np.maximum.accumulate(self.df.strategy_curve.fillna(1)) - np.array(
            self.df.strategy_curve.fillna(1))) / np.maximum.accumulate(self.df.strategy_curve.fillna(1))

        stacked = np.stack((max_drawdowns,) * 3, axis=-1)
        ax.plot(-max_drawdowns, color='#888888', linewidth=0.4)
        ax.fill_between(range(len(max_drawdowns)), -max_drawdowns, 0, facecolor=stacked, alpha=0.2)
        position.plot(range(len(self.df)), self.df.position, color='red')
        position.set_xlim(0, len(self.df))
        if step <= 3:
            step = 3
        position.set_xticks(range(len(self.df))[::step])
        position.set_xticklabels(datetime_index.strftime('%Y-%m-%d')[::step])
        signal.plot(range(len(self.df)), self.df.signal, color='green')
        signal.set_xlim(0, len(self.df))
        signal.set_xticks(range(len(self.df))[::step])
        signal.set_xticklabels(datetime_index.strftime('%Y-%m-%d')[::step])
        ax3.set_xlim(0, len(self.df))
        ax3.set_xticks(range(len(self.df))[::step])
        ax3.set_xticklabels(datetime_index.strftime('%Y-%m-%d')[::step])
        return fig, ax, signal, ax3


class BackTester(object):
    """The Vectorized BackTester class.
    仅支持ALL-IN
    # TODO: 增加position_size
    BackTester is a vectorized backtest for quantitative trading strategies.

    Methods:
        run_():
        optimize_():
    Note:
        1. data在创建时只是浅拷贝, 而在创建回测环境时, 我们进行了一次深拷贝
    """
    __metaclass__ = ABCMeta

    @staticmethod
    def process_strategy(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            self.create_backtest_env()
            if set(self.params).issubset(set(kwargs.keys())):
                for name in self.params:
                    self.params[name] = kwargs[name]
            else:
                for idx, name in enumerate(self.params):
                    self.params[name] = args[1:][idx]

            func(self, *args[1:], **kwargs)
            assert 'signal' in self.backtest_env.columns, "未计算信号"
            assert 'position' in self.backtest_env.columns, "未填充下期持仓信息, 请重新填写"
            assert self.backtest_env['position'].isnull().sum() == 0, "无任何交易, 结束统计"
            # 计算strategy和benchmark的信息
            self.backtest_env['benchmark'] = np.log(
                self.backtest_env['transact_base'] / self.backtest_env['transact_base'].shift(1))
            self.backtest_env['strategy'] = self.backtest_env['position'].shift(1) * self.backtest_env['benchmark']
            self.backtest_env['benchmark_curve'] = self.backtest_env['benchmark'].cumsum().apply(np.exp)
            self.backtest_env['strategy_curve'] = self.backtest_env['strategy'].cumsum().apply(np.exp)

            # 计算交易费用
            self.commission_lst = list()
            if self.buy_commission is not None and self.sell_commission is not None:
                fees_factor = pd.Series(np.nan, index=self.backtest_env.index)
                fees_factor[:] = np.where((self.backtest_env.position - self.backtest_env.position.shift(1)) > 0,
                                          -(self.backtest_env.position - self.backtest_env.position.shift(
                                              1)) * self.buy_commission, np.nan)
                fees_factor[:] = np.where((self.backtest_env.position - self.backtest_env.position.shift(1)) < 0,
                                          (self.backtest_env.position - self.backtest_env.position.shift(
                                              1)) * self.sell_commission, fees_factor)
                fees_factor.fillna(0, inplace=True)
                fees_factor += 1
                # self.commission_lst.append()
                self.fees_factor = fees_factor
                self.backtest_env['strategy_curve'] *= fees_factor.cumprod()
                self.backtest_env['strategy'] = np.log(self.backtest_env['strategy_curve']/self.backtest_env['strategy_curve'].shift(1))
                self.backtest_env['strategy'].fillna(0, inplace=True)
                # 风险评估
            result = dict()
            result['params'] = tuple(self.params.values())
            result['annualized_mean'] = 252 * self.freq_base * self.backtest_env['strategy'].mean()
            result['annualized_std'] = np.sqrt(252 * self.freq_base) * self.backtest_env['strategy'].std()
            if result['annualized_std'] != 0:
                result['sharpe_ratio'] = result['annualized_mean'] / result['annualized_std']
            elif result['annualized_std'] == 0 and result['annualized_mean'] == 0:
                result['sharpe_ratio'] = 0
            elif result['annualized_std'] == 0 and result['annualized_mean'] < 0:
                result['sharpe_ratio'] = -999
            elif result['annualized_std'] == 0 and result['annualized_mean'] > 0:
                result['sharpe_ratio'] = 999
            cummax_value = np.maximum.accumulate(self.backtest_env['strategy_curve'].fillna(1))
            result['max_drawdown'] = np.max((cummax_value - self.backtest_env['strategy_curve'])/cummax_value)
            result['signal_counts'] = np.sum(np.abs(self.backtest_env['signal']))
            return result

        return wrapper

    def __init__(self,
                 symbol_data: pd.DataFrame,
                 transact_base='PreClose',
                #  commissions=(None, None),
                #  commissions=(0.000023, 0.000023),
                 commissions=(0.23, 0.23),
                 slippage_rate=None):

        assert isinstance(symbol_data, pd.DataFrame)
        assert isinstance(symbol_data.index, pd.DatetimeIndex)
        for attr in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert attr in symbol_data.columns

        self.data = symbol_data
        self.freq_base = self.data['Close'].resample('d').apply(lambda x: len(x))[0]
        self.transact_base = transact_base
        self.backtest_env = ...
        self.params = ...
        self.buy_commission = commissions[0]
        self.sell_commission = commissions[1]
        self.slippage_rate = slippage_rate
        self.init()
        self.check_signals = False

    def create_backtest_env(self) -> None:
        self.backtest_env = self.data.copy(deep=True)
        if self.transact_base == 'PreClose':
            self.backtest_env['transact_base'] = self.data['Close']
        elif self.transact_base == 'Open':
            self.backtest_env['transact_base'] = self.data['Open'].shift(-1)
        else:
            raise ValueError(f'transact_base must be "PreClose" or "Open", get {self.transact_base}')
        self.backtest_env['signal'] = np.nan
        self.backtest_env['position'] = np.nan

    @abstractmethod
    def init(self):
        """
        在调用类的构造函数时，自动调用该函数
        """
        self.params = ...
        raise NotImplementedError

    @property
    def params_name(self):
        try:
            return list(self.params.keys())
        except AttributeError:
            self.init()
            return list(self.params.keys())

    @abstractmethod
    @process_strategy.__get__(object)
    def run_(self, *args, **kwargs) -> dict[str: int]:
        """Add the signal and position to the column of the backtest_env.
        and calculate the risk indicators.
        """
        self.backtest_env.position = ...
        raise NotImplementedError("run_ must be implemented")

    def construct_position_(self,
                            keep_raw=False,
                            min_holding_period=None,
                            max_holding_period=None,
                            take_profit=None,
                            stop_loss=None):
        """Modify the position of the backtest_env.
        """
        assert 'signal' in self.backtest_env.columns, '未计算信号'
        self.backtest_env['position'] = self.backtest_env['signal']

        if take_profit is not None and stop_loss is not None:
            mark = pd.Series(np.nan, index=self.backtest_env.index)
            mark[:] = np.where(((self.backtest_env['position'] == 1) +
                                (self.backtest_env['position'] == -1)) > 0, self.backtest_env['transact_base'], np.nan)
            mark.fillna(method='ffill', inplace=True)
            up_band = mark * (1 + take_profit)
            low_band = mark * (1 - stop_loss)
            self.backtest_env['position'] = np.where(self.backtest_env['transact_base'] > up_band, 0,
                                                     self.backtest_env['position'])
            self.backtest_env['position'] = np.where(self.backtest_env['transact_base'] < low_band, 0,
                                                     self.backtest_env['position'])

        if keep_raw:
            self.backtest_env['position'].fillna(0, inplace=True)
        else:
            if max_holding_period is not None:
                self.backtest_env['position'].fillna(method='ffill', limit=max_holding_period, inplace=True)
                self.backtest_env['position'].fillna(0, inplace=True)
            else:
                raise ValueError('max_holding_period should not be None if keep_raw is False')
        self.backtest_env.loc[self.backtest_env.index[0], 'position'] = 0

    def optimize_(self,
                  goal='sharpe_ratio',
                  method='grid',
                  n_jobs=1,
                  **kwargs):
        """
        :param goal: 优化的目标
        :param method:
        :param n_jobs: 进程数
        :return: The best parameters of the backtest_env.
        """
        assert goal in ['annualized_mean', 'annualized_std', 'sharpe_ratio']  # TODO: 增加其它的指标
        for name in self.params:
            assert name in kwargs
            assert isinstance(kwargs[name], abc.Iterable)

        temp = itertools.product(*[kwargs[x] for x in self.params])
        if method == 'grid':
            if n_jobs > 1:
                print('调用并行')
                with multiprocessing.Pool(n_jobs) as p:
                    results = p.starmap(self.run_, temp)
            else:
                print('不调用并行')
                results = [self.run_(*args) for args in temp]
            rlt = max(results, key=lambda x: x[goal])
            return rlt

    def summary(self, *args, **kwargs) -> Displayer:
        return Displayer(self.backtest_env)

    def clear(self):
        del self.backtest_env

    @staticmethod
    def cross_up(series1, series2):
        assert isinstance(series1, pd.Series)
        assert isinstance(series2, pd.Series)
        return (series1 > series2) * (series1.shift(1) < series2.shift(1))

    @staticmethod
    def cross_down(series1, series2):
        assert isinstance(series1, pd.Series)
        assert isinstance(series2, pd.Series)
        return (series1 < series2) * (series1.shift(1) > series2.shift(1))
