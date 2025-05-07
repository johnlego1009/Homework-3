"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, momentum_lookback=100, top_n=5, rebalance_period=15):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        # self.gamma = gamma
        self.momentum_lookback = momentum_lookback  # === 新增：動能回測期 ===
        self.top_n = top_n  # === 新增：選擇前 N 名動能資產 ===
        self.rebalance_period = rebalance_period  # === 新增：再平衡週期 ===

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        for i in range(max(self.lookback, self.momentum_lookback), len(self.price), self.rebalance_period):
            date = self.price.index[i]

            # === Step 1: 動能排名 ===
            momentum = (
                self.price[assets].iloc[i - self.momentum_lookback:i].iloc[-1]
                / self.price[assets].iloc[i - self.momentum_lookback:i].iloc[0]
                - 1
            )
            # 取動能最強的 top_n 資產
            top_assets = momentum.nlargest(self.top_n).index.tolist()

            # === Step 2: ERC（風險貢獻平價）計算前 N 名資產的權重 ===
            R_n = self.returns[top_assets].iloc[i - self.lookback:i]
            weights = self.erc_weights(R_n)

            self.portfolio_weights.loc[date, top_assets] = weights

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

        # 進行權重歸一化，確保不會有槓桿
        self.portfolio_weights = self.portfolio_weights.div(
            self.portfolio_weights.sum(axis=1).clip(lower=1e-10), axis=0
        )

    def erc_weights(self, R_n):
        # 計算風險貢獻平價（ERC）權重
        cov = R_n.cov().values
        n = len(cov)

        # 計算每個資產對於投資組合總風險的貢獻
        def risk_contribution(weights):
            port_var = weights.T @ cov @ weights
            mrc = cov @ weights
            rc = weights * mrc
            return rc / port_var

        # 目標函數：最小化風險貢獻的偏差
        def objective(weights):
            rc = risk_contribution(weights)
            return ((rc - rc.mean()) ** 2).sum()

        # 初始猜測權重：均等分配
        x0 = np.ones(n) / n
        bounds = [(0, 1) for _ in range(n)]  # 權重限制在 0 到 1 之間
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # 權重總和為 1

        # 使用 scipy 的 minimize 進行優化
        from scipy.optimize import minimize
        res = minimize(objective, x0, bounds=bounds, constraints=constraints)

        if res.success:
            # 若優化成功，將權重限制在 0 到 1 範圍內，並進行歸一化
            weights = np.clip(res.x, 0, 1)
            return (weights / weights.sum()).tolist()
        else:
            # 若優化失敗，均等分配權重
            return [1 / n] * n

    def calculate_portfolio_returns(self):
        # 確保權重已經計算
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # 計算投資組合回報
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]

        # 原始投資組合回報
        raw_returns = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

        # 波動率平滑（過去 20 日）
        volatility = raw_returns.rolling(window=20).std()

        # 目標年化波動率（例如 0.15 = 15%）
        target_vol = 0.15
        scaling = target_vol / (volatility * np.sqrt(252))
        scaling = scaling.clip(upper=2)  # 限制槓桿上限，防止過度放大

        # 計算最終的投資組合回報
        self.portfolio_returns["Portfolio"] = raw_returns * scaling

        # 填補缺失的回報數據
        self.portfolio_returns.fillna(0, inplace=True)

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)
