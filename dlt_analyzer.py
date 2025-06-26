# -*- coding: utf-8 -*-
"""
大乐透彩票数据分析与推荐系统
================================

本脚本整合了统计分析、机器学习和策略化组合生成，为大乐透彩票提供数据驱动的
号码推荐。脚本支持两种运行模式，由全局变量 `ENABLE_OPTUNA_OPTIMIZATION` 控制：

1.  **分析模式 (默认 `False`)**:
    使用内置的 `DEFAULT_WEIGHTS` 权重，基于滑动窗口数据进行频率/模式/关联分析、
    策略回测，并为下一期生成推荐号码。所有结果会输出到一个带时间戳的详细报告文件中。

2.  **优化模式 (`True`)**:
    在分析前，首先运行 Optuna 框架进行参数搜索，以找到在近期历史数据上
    表现最佳的一组权重。然后，自动使用这组优化后的权重来完成后续的分析、
    回测和推荐。优化过程和结果也会记录在报告中。

**数据窗口策略**:
- 频率分析：使用最近50期数据（避免历史热号偏向）
- 模式分析：使用最近100期数据（捕捉中期模式）
- 关联规则：使用最近200期数据（确保规则挖掘的样本充足）
- 机器学习：使用全部历史数据（保证模型训练的样本量）

版本: 5.1 (大乐透适配版)
"""

# --- 标准库导入 ---
import os
import sys
import json
import time
import datetime
import logging
import io
import random
from collections import Counter
from contextlib import redirect_stdout
from typing import (Union, Optional, List, Dict, Tuple, Any)
from functools import partial

# --- 第三方库导入 ---
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import concurrent.futures

# ==============================================================================
# --- 全局常量与配置 ---
# ==============================================================================

# --------------------------
# --- 路径与模式配置 ---
# --------------------------
# 脚本文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始大乐透数据CSV文件路径 (由 dlt_data_processor.py 生成)
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'daletou.csv')
# 预处理后的数据缓存文件路径，避免每次都重新计算特征
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'daletou_processed.csv')

# 运行模式配置:
# True  -> 运行参数优化，耗时较长，但可能找到更优策略。
# False -> 使用默认权重进行快速分析和推荐。
ENABLE_OPTUNA_OPTIMIZATION = True

# --------------------------
# --- 策略开关配置 ---
# --------------------------
# 是否启用最终推荐组合层面的"反向思维"策略 (移除得分最高的几注)
ENABLE_FINAL_COMBO_REVERSE = True
# 在启用反向思维并移除组合后，是否从候选池中补充新的组合以达到目标数量
ENABLE_REVERSE_REFILL = True

# --------------------------
# --- 彩票规则配置 ---
# --------------------------
# 红球的有效号码范围 (1到35) - 大乐透规则
RED_BALL_RANGE = range(1, 36)
# 蓝球的有效号码范围 (1到12) - 大乐透规则
BLUE_BALL_RANGE = range(1, 13)
# 红球三分区定义，用于特征工程和模式分析
RED_ZONES = {'Zone1': (1, 12), 'Zone2': (13, 24), 'Zone3': (25, 35)}

# --------------------------
# --- 分析与执行参数配置 ---
# --------------------------
# 机器学习模型使用的滞后特征阶数 (e.g., 使用前1、3、5、10期的数据作为特征)
ML_LAG_FEATURES = [1, 3, 5, 8,13]
# 用于生成乘积交互特征的特征对 (e.g., 红球和值 * 红球奇数个数)
ML_INTERACTION_PAIRS = [('red_sum', 'red_odd_count')]
# 用于生成自身平方交互特征的特征 (e.g., 红球跨度的平方)
ML_INTERACTION_SELF = ['red_span']
# 计算号码"近期"出现频率时所参考的期数窗口大小
RECENT_FREQ_WINDOW = 50
# 计算主要频率分析时所参考的期数窗口大小 (用于替代全历史数据)
MAIN_FREQ_WINDOW = 50
# 关联规则挖掘和模式分析的窗口大小
ASSOCIATION_ANALYSIS_WINDOW = 50
PATTERN_ANALYSIS_WINDOW = 100
# 在分析模式下，进行策略回测时所评估的总期数
BACKTEST_PERIODS_COUNT = 100
# 在优化模式下，每次试验用于快速评估性能的回测期数 (数值越小优化越快)
OPTIMIZATION_BACKTEST_PERIODS = 20
# 在优化模式下，Optuna 进行参数搜索的总试验次数
OPTIMIZATION_TRIALS = 100
# 训练机器学习模型时，一个球号在历史数据中至少需要出现的次数 (防止样本过少导致模型不可靠)
MIN_POSITIVE_SAMPLES_FOR_ML = 25

# ==============================================================================
# --- 默认权重配置 (这些参数可被Optuna优化) ---
# ==============================================================================
# 这里的每一项都是一个可调整的策略参数，共同决定了最终的推荐结果。
DEFAULT_WEIGHTS = {
    # --- 反向思维 ---
    # 若启用反向思维，从最终推荐列表中移除得分最高的组合的比例
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.3,

    # --- 组合生成 ---
    # 最终向用户推荐的组合（注数）数量
    'NUM_COMBINATIONS_TO_GENERATE': 10,
    # 构建红球候选池时，从所有红球中选取分数最高的N个
    'TOP_N_RED_FOR_CANDIDATE': 20,  # 大乐透红球数量调整
    # 构建蓝球候选池时，从所有蓝球中选取分数最高的N个
    'TOP_N_BLUE_FOR_CANDIDATE': 8,  # 大乐透蓝球数量调整

    # --- 红球评分权重 ---
    # 红球历史总频率得分的权重
    'FREQ_SCORE_WEIGHT': 28.19,
    # 红球当前遗漏值（与平均遗漏的偏差）得分的权重
    'OMISSION_SCORE_WEIGHT': 19.92,
    # 红球当前遗漏与其历史最大遗漏比率的得分权重
    'MAX_OMISSION_RATIO_SCORE_WEIGHT_RED': 16.12,
    # 红球近期出现频率的得分权重
    'RECENT_FREQ_SCORE_WEIGHT_RED': 15.71,
    # 红球的机器学习模型预测出现概率的得分权重
    'ML_PROB_SCORE_WEIGHT_RED': 22.43,

    # --- 蓝球评分权重 ---
    # 蓝球历史总频率得分的权重
    'BLUE_FREQ_SCORE_WEIGHT': 27.11,
    # 蓝球当前遗漏值（与平均遗漏的偏差）得分的权重
    'BLUE_OMISSION_SCORE_WEIGHT': 23.26,
    # 蓝球的机器学习模型预测出现概率的得分权重
    'ML_PROB_SCORE_WEIGHT_BLUE': 43.48,

    # --- 组合属性匹配奖励 ---
    # 推荐组合的红球奇数个数若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 13.10,
    # 推荐组合的蓝球奇偶性若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_BLUE_ODD_MATCH_BONUS': 0.40,
    # 推荐组合的红球区间分布若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_ZONE_MATCH_BONUS': 13.12,
    # 推荐组合的蓝球大小若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_BLUE_SIZE_MATCH_BONUS': 0.84,

    # --- 关联规则挖掘(ARM)参数与奖励 ---
    # ARM算法的最小支持度阈值
    'ARM_MIN_SUPPORT': 0.01,
    # ARM算法的最小置信度阈值
    'ARM_MIN_CONFIDENCE': 0.53,
    # ARM算法的最小提升度阈值
    'ARM_MIN_LIFT': 1.53,
    # 推荐组合若命中了某条挖掘出的关联规则，其获得的基础奖励分值
    'ARM_COMBINATION_BONUS_WEIGHT': 18.86,
    # 在计算ARM奖励时，规则的提升度(lift)对此奖励的贡献乘数因子
    'ARM_BONUS_LIFT_FACTOR': 0.48,
    # 在计算ARM奖励时，规则的置信度(confidence)对此奖励的贡献乘数因子
    'ARM_BONUS_CONF_FACTOR': 0.25,

    # --- 组合多样性控制 ---
    # 最终推荐的任意两注组合之间，其红球号码至少要有几个是不同的
    'DIVERSITY_MIN_DIFFERENT_REDS': 2,  # 大乐透调整为2个
    
    # --- 热冷号码策略权重 (基于经验规律) ---
    # 最近5期内出现过的号码权重加成 (热号策略)
    'HOT_NUMBERS_5_PERIODS_WEIGHT': 25.0,
    # 最近6期内没有出现过的号码权重加成 (冷号策略)  
    'COLD_NUMBERS_6_PERIODS_WEIGHT': 20.0,
    # 蓝球热号策略权重
    'BLUE_HOT_NUMBERS_5_PERIODS_WEIGHT': 15.0,
    # 蓝球冷号策略权重
    'BLUE_COLD_NUMBERS_6_PERIODS_WEIGHT': 12.0,
}

# ==============================================================================
# --- 机器学习模型参数配置 ---
# ==============================================================================
# 这些是 LightGBM 机器学习模型的核心超参数。
LGBM_PARAMS = {
    'objective': 'binary',              # 目标函数：二分类问题（预测一个球号是否出现）
    'boosting_type': 'gbdt',            # 提升类型：梯度提升决策树
    'learning_rate': 0.06,              # 学习率：控制每次迭代的步长 (优化后)
    'n_estimators': 200,                # 树的数量：总迭代次数 (优化后)
    'num_leaves': 30,                   # 每棵树的最大叶子节点数：控制模型复杂度 (优化后)
    'min_child_samples': 12,            # 一个叶子节点上所需的最小样本数：防止过拟合 (优化后)
    'lambda_l1': 0.15,                  # L1 正则化
    'lambda_l2': 0.15,                  # L2 正则化
    'feature_fraction': 0.85,           # 特征采样比例：每次迭代随机选择85%的特征 (优化后)
    'bagging_fraction': 0.9,            # 数据采样比例：每次迭代随机选择90%的数据 (优化后)
    'bagging_freq': 5,                  # 数据采样的频率：每5次迭代进行一次
    'seed': 42,                         # 随机种子：确保结果可复现
    'n_jobs': 1,                        # 并行线程数：设为1以在多进程环境中避免冲突
    'verbose': -1,                      # 控制台输出级别：-1表示静默
}

# ==============================================================================
# --- 日志系统配置 ---
# ==============================================================================
# 创建两种格式化器
console_formatter = logging.Formatter('%(message)s')  # 用于控制台的简洁格式
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s') # 用于文件的详细格式

# 主日志记录器
logger = logging.getLogger('dlt_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False # 防止日志向根记录器传递

# 进度日志记录器 (用于回测和Optuna进度条，避免被详细格式污染)
progress_logger = logging.getLogger('dlt_progress')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False

# 设置全局控制台处理器（将在主程序中被配置）
global_console_handler = None

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """设置控制台日志的详细程度和格式。"""
    global global_console_handler
    if global_console_handler:
        logger.removeHandler(global_console_handler)
        progress_logger.removeHandler(global_console_handler)
    
    formatter = console_formatter if use_simple_formatter else detailed_formatter
    global_console_handler = logging.StreamHandler()
    global_console_handler.setLevel(level)
    global_console_handler.setFormatter(formatter)
    
    logger.addHandler(global_console_handler)
    progress_logger.addHandler(global_console_handler)

class SuppressOutput:
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self.old_stdout:
            sys.stdout.close()
            sys.stdout = self.old_stdout
        return False

def get_prize_level(red_hits: int, blue_hits: int) -> Optional[str]:
    """
    根据红球和蓝球命中数，返回大乐透的奖级名称。
    
    官方大乐透奖级规则：
    一等奖：5+2
    二等奖：5+1  
    三等奖：5+0
    四等奖：4+2
    五等奖：4+1
    六等奖：3+2
    七等奖：4+0
    八等奖：3+1 或 2+2
    九等奖：3+0 或 1+2 或 2+1 或 0+2
    """
    if red_hits == 5 and blue_hits == 2:
        return "一等奖"
    elif red_hits == 5 and blue_hits == 1:
        return "二等奖"
    elif red_hits == 5 and blue_hits == 0:
        return "三等奖"
    elif red_hits == 4 and blue_hits == 2:
        return "四等奖"
    elif red_hits == 4 and blue_hits == 1:
        return "五等奖"
    elif red_hits == 3 and blue_hits == 2:
        return "六等奖"
    elif red_hits == 4 and blue_hits == 0:
        return "七等奖"
    elif (red_hits == 3 and blue_hits == 1) or (red_hits == 2 and blue_hits == 2):
        return "八等奖"
    elif (red_hits == 3 and blue_hits == 0) or (red_hits == 1 and blue_hits == 2) or (red_hits == 2 and blue_hits == 1) or (red_hits == 0 and blue_hits == 2):
        return "九等奖"
    else:
        return None

def format_time(seconds: float) -> str:
    """将秒数格式化为易读的时间字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    从CSV文件加载大乐透数据。
    
    Args:
        file_path (str): CSV文件路径
        
    Returns:
        Optional[pd.DataFrame]: 成功时返回DataFrame，失败时返回None
    """
    if not os.path.exists(file_path):
        logger.error(f"数据文件不存在: {file_path}")
        return None
    
    try:
        # 支持多种编码格式的文件读取
        encodings = ['utf-8', 'gbk', 'latin-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"成功使用 {encoding} 编码加载数据，共 {len(df)} 行记录。")
                return df
            except UnicodeDecodeError:
                continue
        
        logger.error(f"无法使用任何支持的编码格式读取文件: {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"加载数据时发生错误: {e}")
        return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    清理和结构化大乐透数据。
    
    Args:
        df (pd.DataFrame): 原始数据DataFrame
        
    Returns:
        Optional[pd.DataFrame]: 清理后的DataFrame，失败时返回None
    """
    try:
        logger.info("开始数据清理和结构化...")
        
        # 重命名列以确保一致性
        if '期号' not in df.columns or '红球' not in df.columns or '蓝球' not in df.columns:
            logger.error("数据文件缺少必要的列：期号、红球、蓝球")
            return None
        
        # 创建副本避免修改原始数据
        df_clean = df.copy()
        
        # 清理期号
        df_clean['期号'] = pd.to_numeric(df_clean['期号'], errors='coerce')
        df_clean = df_clean.dropna(subset=['期号'])
        df_clean['期号'] = df_clean['期号'].astype(int)
        
        # 解析红球和蓝球
        red_balls_list = []
        blue_balls_list = []
        
        valid_rows = []
        for idx, row in df_clean.iterrows():
            try:
                # 解析红球（5个）
                red_str = str(row['红球']).strip()
                red_balls = [int(x.strip()) for x in red_str.split(',')]
                
                # 解析蓝球（2个）
                blue_str = str(row['蓝球']).strip()
                blue_balls = [int(x.strip()) for x in blue_str.split(',')]
                
                # 验证数量和范围
                if (len(red_balls) == 5 and len(blue_balls) == 2 and
                    all(1 <= r <= 35 for r in red_balls) and
                    all(1 <= b <= 12 for b in blue_balls)):
                    
                    red_balls_list.append(sorted(red_balls))
                    blue_balls_list.append(sorted(blue_balls))
                    valid_rows.append(idx)
                else:
                    logger.warning(f"跳过无效数据行 {idx}: 红球={red_balls}, 蓝球={blue_balls}")
                    
            except (ValueError, AttributeError) as e:
                logger.warning(f"跳过解析失败的行 {idx}: {e}")
                continue
        
        if not valid_rows:
            logger.error("没有找到有效的开奖数据。")
            return None
        
        # 重建DataFrame
        df_final = df_clean.loc[valid_rows].copy()
        
        # 添加结构化的红球和蓝球列
        for i in range(5):
            df_final[f'红球_{i+1}'] = [balls[i] for balls in red_balls_list]
        
        for i in range(2):
            df_final[f'蓝球_{i+1}'] = [balls[i] for balls in blue_balls_list]
        
        # 按期号排序
        df_final = df_final.sort_values('期号').reset_index(drop=True)
        
        logger.info(f"数据清理完成，有效记录: {len(df_final)} 条")
        return df_final
        
    except Exception as e:
        logger.error(f"数据清理过程中发生错误: {e}")
        return None

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    为大乐透数据创建特征工程。
    
    Args:
        df (pd.DataFrame): 清理后的数据DataFrame
        
    Returns:
        Optional[pd.DataFrame]: 特征工程后的DataFrame，失败时返回None
    """
    try:
        logger.info("开始特征工程...")
        
        df_features = df.copy()
        
        # 基础特征 - 红球
        df_features['red_sum'] = df_features[['红球_1', '红球_2', '红球_3', '红球_4', '红球_5']].sum(axis=1)
        df_features['red_span'] = df_features['红球_5'] - df_features['红球_1']
        df_features['red_odd_count'] = df_features[['红球_1', '红球_2', '红球_3', '红球_4', '红球_5']].apply(lambda x: (x % 2).sum(), axis=1)
        df_features['red_even_count'] = 5 - df_features['red_odd_count']
        
        # 基础特征 - 蓝球
        df_features['blue_sum'] = df_features[['蓝球_1', '蓝球_2']].sum(axis=1)
        df_features['blue_span'] = df_features['蓝球_2'] - df_features['蓝球_1']
        df_features['blue_odd_count'] = df_features[['蓝球_1', '蓝球_2']].apply(lambda x: (x % 2).sum(), axis=1)
        df_features['blue_even_count'] = 2 - df_features['blue_odd_count']
        
        # 区间分析 - 红球（重新调整为大乐透的35个号码）
        for zone, (start, end) in RED_ZONES.items():
            zone_count = 0
            for i in range(1, 6):  # 5个红球
                zone_count += ((df_features[f'红球_{i}'] >= start) & (df_features[f'红球_{i}'] <= end)).astype(int)
            df_features[f'red_{zone.lower()}_count'] = zone_count
        
        # 大小球分析 - 蓝球（1-6为小，7-12为大）
        df_features['blue_big_count'] = ((df_features['蓝球_1'] > 6).astype(int) + 
                                        (df_features['蓝球_2'] > 6).astype(int))
        df_features['blue_small_count'] = 2 - df_features['blue_big_count']
        
        logger.info(f"特征工程完成，新增 {len(df_features.columns) - len(df.columns)} 个特征")
        return df_features
        
    except Exception as e:
        logger.error(f"特征工程过程中发生错误: {e}")
        return None

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """
    分析大乐透号码的频率和遗漏情况。
    
    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame
        
    Returns:
        dict: 包含频率和遗漏分析结果的字典
    """
    try:
        logger.info("开始频率和遗漏分析...")
        
        result = {}
        
        # 使用最近N期数据进行主要分析，而非全部历史数据
        analysis_window = df.tail(MAIN_FREQ_WINDOW) if len(df) >= MAIN_FREQ_WINDOW else df
        logger.info(f"使用最近{len(analysis_window)}期数据进行频率分析 (窗口大小: {MAIN_FREQ_WINDOW}期)")
        
        # 红球分析
        red_freq = {}
        red_omission = {}
        red_max_omission = {}
        
        for ball_num in RED_BALL_RANGE:
            # 计算滑动窗口内的频率
            appearances = []
            for i in range(1, 6):  # 5个红球
                appearances.extend(analysis_window[analysis_window[f'红球_{i}'] == ball_num].index.tolist())
            
            red_freq[ball_num] = len(appearances)
            
            # 计算当前遗漏和最大遗漏（基于分析窗口）
            if appearances:
                # 当前遗漏基于分析窗口的最后一期
                current_omission = len(analysis_window) - 1 - (max(appearances) - analysis_window.index[0])
                red_omission[ball_num] = current_omission
                
                # 计算最大遗漏（基于分析窗口内的间隔）
                gaps = []
                prev_idx = -1
                window_relative_appearances = [idx - analysis_window.index[0] for idx in sorted(appearances)]
                for idx in window_relative_appearances:
                    if prev_idx >= 0:
                        gaps.append(idx - prev_idx - 1)
                    prev_idx = idx
                gaps.append(current_omission)
                
                red_max_omission[ball_num] = max(gaps) if gaps else current_omission
            else:
                red_omission[ball_num] = len(analysis_window)
                red_max_omission[ball_num] = len(analysis_window)
        
        # 蓝球分析
        blue_freq = {}
        blue_omission = {}
        blue_max_omission = {}
        
        for ball_num in BLUE_BALL_RANGE:
            # 计算滑动窗口内的频率
            appearances = []
            for i in range(1, 3):  # 2个蓝球
                appearances.extend(analysis_window[analysis_window[f'蓝球_{i}'] == ball_num].index.tolist())
            
            blue_freq[ball_num] = len(appearances)
            
            # 计算当前遗漏和最大遗漏（基于分析窗口）
            if appearances:
                # 当前遗漏基于分析窗口的最后一期
                current_omission = len(analysis_window) - 1 - (max(appearances) - analysis_window.index[0])
                blue_omission[ball_num] = current_omission
                
                # 计算最大遗漏（基于分析窗口内的间隔）
                gaps = []
                prev_idx = -1
                window_relative_appearances = [idx - analysis_window.index[0] for idx in sorted(appearances)]
                for idx in window_relative_appearances:
                    if prev_idx >= 0:
                        gaps.append(idx - prev_idx - 1)
                    prev_idx = idx
                gaps.append(current_omission)
                
                blue_max_omission[ball_num] = max(gaps) if gaps else current_omission
            else:
                blue_omission[ball_num] = len(analysis_window)
                blue_max_omission[ball_num] = len(analysis_window)
        
        # 计算平均间隔和近期频率
        red_avg_interval = {}
        blue_avg_interval = {}
        recent_red_freq = {}
        recent_blue_freq = {}
        
        # 红球平均间隔和近期频率
        for ball_num in RED_BALL_RANGE:
            if red_freq[ball_num] > 0:
                red_avg_interval[ball_num] = len(analysis_window) / red_freq[ball_num]
            else:
                red_avg_interval[ball_num] = len(analysis_window)
            
            # 近期频率 (最近RECENT_FREQ_WINDOW期)
            recent_window = df.tail(RECENT_FREQ_WINDOW) if len(df) >= RECENT_FREQ_WINDOW else df
            recent_appearances = 0
            for i in range(1, 6):  # 5个红球
                recent_appearances += len(recent_window[recent_window[f'红球_{i}'] == ball_num])
            recent_red_freq[ball_num] = recent_appearances
        
        # 蓝球平均间隔和近期频率
        for ball_num in BLUE_BALL_RANGE:
            if blue_freq[ball_num] > 0:
                blue_avg_interval[ball_num] = len(analysis_window) / blue_freq[ball_num]
            else:
                blue_avg_interval[ball_num] = len(analysis_window)
            
            # 近期频率 (最近RECENT_FREQ_WINDOW期)
            recent_window = df.tail(RECENT_FREQ_WINDOW) if len(df) >= RECENT_FREQ_WINDOW else df
            recent_appearances = 0
            for i in range(1, 3):  # 2个蓝球
                recent_appearances += len(recent_window[recent_window[f'蓝球_{i}'] == ball_num])
            recent_blue_freq[ball_num] = recent_appearances
        
        # 热冷号码分析：最近5期热号 + 6期内冷号策略
        red_hot_5_periods = {}  # 最近5期内出现过的红球
        red_cold_6_periods = {}  # 最近6期内没有出现过的红球
        blue_hot_5_periods = {}  # 最近5期内出现过的蓝球
        blue_cold_6_periods = {}  # 最近6期内没有出现过的蓝球
        
        # 分析红球热冷号码
        recent_5_periods = df.tail(5) if len(df) >= 5 else df
        recent_6_periods = df.tail(6) if len(df) >= 6 else df
        
        for ball_num in RED_BALL_RANGE:
            # 检查是否在最近5期内出现过
            appeared_in_5 = False
            for i in range(1, 6):  # 5个红球
                if len(recent_5_periods[recent_5_periods[f'红球_{i}'] == ball_num]) > 0:
                    appeared_in_5 = True
                    break
            red_hot_5_periods[ball_num] = 1 if appeared_in_5 else 0
            
            # 检查是否在最近6期内没有出现过
            appeared_in_6 = False
            for i in range(1, 6):  # 5个红球
                if len(recent_6_periods[recent_6_periods[f'红球_{i}'] == ball_num]) > 0:
                    appeared_in_6 = True
                    break
            red_cold_6_periods[ball_num] = 1 if not appeared_in_6 else 0
        
        # 分析蓝球热冷号码
        for ball_num in BLUE_BALL_RANGE:
            # 检查是否在最近5期内出现过
            appeared_in_5 = False
            for i in range(1, 3):  # 2个蓝球
                if len(recent_5_periods[recent_5_periods[f'蓝球_{i}'] == ball_num]) > 0:
                    appeared_in_5 = True
                    break
            blue_hot_5_periods[ball_num] = 1 if appeared_in_5 else 0
            
            # 检查是否在最近6期内没有出现过
            appeared_in_6 = False
            for i in range(1, 3):  # 2个蓝球
                if len(recent_6_periods[recent_6_periods[f'蓝球_{i}'] == ball_num]) > 0:
                    appeared_in_6 = True
                    break
            blue_cold_6_periods[ball_num] = 1 if not appeared_in_6 else 0

        # 构建包含所有必要数据的结果字典
        result = {
            'red_freq': red_freq,
            'red_omission': red_omission,
            'red_max_omission': red_max_omission,
            'blue_freq': blue_freq,
            'blue_omission': blue_omission,
            'blue_max_omission': blue_max_omission,
            # 修复缺失的键名
            'current_omission': red_omission,  # 当前遗漏（红球）
            'average_interval': red_avg_interval,  # 平均间隔（红球）
            'max_historical_omission_red': red_max_omission,  # 最大历史遗漏（红球）
            'recent_N_freq_red': recent_red_freq,  # 近期频率（红球）
            'blue_current_omission': blue_omission,  # 当前遗漏（蓝球）
            'blue_average_interval': blue_avg_interval,  # 平均间隔（蓝球）
            'recent_N_freq_blue': recent_blue_freq,  # 近期频率（蓝球）
            # 新增热冷号码分析
            'red_hot_5_periods': red_hot_5_periods,  # 红球5期热号
            'red_cold_6_periods': red_cold_6_periods,  # 红球6期冷号
            'blue_hot_5_periods': blue_hot_5_periods,  # 蓝球5期热号
            'blue_cold_6_periods': blue_cold_6_periods,  # 蓝球6期冷号
        }
        
        logger.info("频率和遗漏分析完成")
        return result
        
    except Exception as e:
        logger.error(f"频率和遗漏分析过程中发生错误: {e}")
        return {}

def analyze_patterns(df: pd.DataFrame) -> dict:
    """
    分析历史数据中的常见模式，如最常见的和值、奇偶比、区间分布等。

    Args:
        df (pd.DataFrame): 包含特征工程后历史数据的DataFrame。

    Returns:
        dict: 包含最常见模式的字典。
    """
    if df is None or df.empty: return {}
    
    # 使用最近N期数据进行模式分析，而非全部历史数据
    pattern_window = df.tail(PATTERN_ANALYSIS_WINDOW) if len(df) >= PATTERN_ANALYSIS_WINDOW else df
    logger.info(f"使用最近{len(pattern_window)}期数据进行模式分析 (窗口大小: {PATTERN_ANALYSIS_WINDOW}期)")
    
    res = {}
    def safe_mode(s): return s.mode().iloc[0] if not s.empty and not s.mode().empty else None
    
    for col, name in [('red_sum', 'sum'), ('red_span', 'span'), ('red_odd_count', 'odd_count')]:
        if col in pattern_window.columns: res[f'most_common_{name}'] = safe_mode(pattern_window[col])
        
    zone_cols = [f'red_{zone.lower()}_count' for zone in RED_ZONES.keys()]
    if all(c in pattern_window.columns for c in zone_cols):
        dist_counts = pattern_window[zone_cols].apply(tuple, axis=1).value_counts()
        if not dist_counts.empty: res['most_common_zone_distribution'] = dist_counts.index[0]
        
    if 'blue_odd_count' in pattern_window.columns: res['most_common_blue_is_odd'] = safe_mode(pattern_window['blue_odd_count'] > 0)
    if 'blue_big_count' in pattern_window.columns: res['most_common_blue_is_large'] = safe_mode(pattern_window['blue_big_count'] > 0)
    
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """
    使用Apriori算法挖掘红球号码之间的关联规则（例如，哪些号码倾向于一起出现）。

    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。
        weights_config (Dict): 包含ARM算法参数(min_support, min_confidence, min_lift)的字典。

    Returns:
        pd.DataFrame: 一个包含挖掘出的强关联规则的DataFrame。
    """
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.01)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.5)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.5)
    red_cols = [f'红球_{i}' for i in range(1, 6)]  # 大乐透5个红球
    if df is None or df.empty: return pd.DataFrame()
    
    # 使用最近N期数据进行关联规则挖掘，而非全部历史数据
    association_window = df.tail(ASSOCIATION_ANALYSIS_WINDOW) if len(df) >= ASSOCIATION_ANALYSIS_WINDOW else df
    logger.info(f"使用最近{len(association_window)}期数据进行关联规则分析 (窗口大小: {ASSOCIATION_ANALYSIS_WINDOW}期)")
    
    try:
        transactions = association_window[red_cols].astype(str).values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_oh = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_oh, min_support=min_s, use_colnames=True)
        if frequent_itemsets.empty: return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_l)
        strong_rules = rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        return strong_rules
        
    except Exception as e:
        logger.error(f"关联规则分析失败: {e}"); return pd.DataFrame()

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """
    为机器学习模型创建滞后特征（将历史期的特征作为当前期的输入）和交互特征。

    Args:
        df (pd.DataFrame): 包含基础特征的DataFrame。
        lags (List[int]): 滞后阶数列表, e.g., [1, 3, 5]。

    Returns:
        Optional[pd.DataFrame]: 一个只包含滞后和交互特征的DataFrame。
    """
    if df is None or df.empty or not lags: return None
    
    feature_cols = [col for col in df.columns if 'red_' in col or 'blue_' in col]
    df_features = df[feature_cols].copy()
    
    # 创建交互特征
    for c1, c2 in ML_INTERACTION_PAIRS:
        if c1 in df_features and c2 in df_features: df_features[f'{c1}_x_{c2}'] = df_features[c1] * df_features[c2]
    for c in ML_INTERACTION_SELF:
        if c in df_features: df_features[f'{c}_sq'] = df_features[c]**2
        
    # 创建滞后特征
    all_feature_cols = df_features.columns.tolist()
    lagged_dfs = [df_features[all_feature_cols].shift(lag).add_suffix(f'_lag{lag}') for lag in lags]
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df.dropna(inplace=True)
    
    return final_df if not final_df.empty else None

def train_single_lgbm_model(ball_type_str: str, ball_number: int, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[LGBMClassifier], Optional[str]]:
    """为单个球号训练一个LGBM二分类模型（预测它是否出现）。"""
    if y_train.sum() < MIN_POSITIVE_SAMPLES_FOR_ML or y_train.nunique() < 2:
        return None, None # 样本不足或只有一类，无法训练
        
    model_key = f'lgbm_{ball_number}'
    model_params = LGBM_PARAMS.copy()
    
    # 类别不平衡处理：给样本量较少的类别（中奖）更高的权重
    if (pos_count := y_train.sum()) > 0:
        model_params['scale_pos_weight'] = (len(y_train) - pos_count) / pos_count
        
    try:
        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        return model, model_key
    except Exception as e:
        logger.debug(f"训练LGBM for {ball_type_str} {ball_number} 失败: {e}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int]) -> Optional[Dict[str, Any]]:
    """为所有红球和蓝球并行训练预测模型。"""
    if (X := create_lagged_features(df_train_raw.copy(), ml_lags_list)) is None or X.empty:
        logger.warning("创建滞后特征失败或结果为空，跳过模型训练。")
        return None
        
    if (target_df := df_train_raw.loc[X.index].copy()).empty: return None
    
    red_cols = [f'红球_{i}' for i in range(1, 6)]  # 大乐透5个红球
    trained_models = {'red': {}, 'blue': {}, 'feature_cols': X.columns.tolist()}
    
    # 使用进程池并行训练，加快速度
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        # 为每个红球提交训练任务
        for ball_num in RED_BALL_RANGE:
            y = target_df[red_cols].eq(ball_num).any(axis=1).astype(int)
            future = executor.submit(train_single_lgbm_model, '红球', ball_num, X, y)
            futures[future] = ('red', ball_num)
        # 为每个蓝球提交训练任务
        for ball_num in BLUE_BALL_RANGE:
            y = target_df[[f'蓝球_{i}' for i in range(1, 3)]].eq(ball_num).any(axis=1).astype(int)  # 大乐透2个蓝球
            future = executor.submit(train_single_lgbm_model, '蓝球', ball_num, X, y)
            futures[future] = ('blue', ball_num)
            
        for future in concurrent.futures.as_completed(futures):
            ball_type, ball_num = futures[future]
            try:
                model, model_key = future.result()
                if model and model_key:
                    trained_models[ball_type][model_key] = model
            except Exception as e:
                logger.error(f"训练球号 {ball_num} ({ball_type}) 的模型时出现异常: {e}")

    return trained_models if trained_models['red'] or trained_models['blue'] else None

def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[Dict], ml_lags_list: List[int]) -> Dict[str, Dict[int, float]]:
    """使用训练好的模型预测下一期每个号码的出现概率。"""
    probs = {'red': {}, 'blue': {}}
    if not trained_models or not (feat_cols := trained_models.get('feature_cols')):
        return probs
        
    max_lag = max(ml_lags_list) if ml_lags_list else 0
    if len(df_historical) < max_lag + 1:
        return probs # 数据不足以创建预测所需的特征
        
    if (predict_X := create_lagged_features(df_historical.tail(max_lag + 1), ml_lags_list)) is None:
        return probs
        
    predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
    
    for ball_type, ball_range in [('red', RED_BALL_RANGE), ('blue', BLUE_BALL_RANGE)]:
        for ball_num in ball_range:
            if (model := trained_models.get(ball_type, {}).get(f'lgbm_{ball_num}')):
                try:
                    # 预测类别为1（出现）的概率
                    probs[ball_type][ball_num] = model.predict_proba(predict_X)[0, 1]
                except Exception:
                    pass
    return probs

def calculate_scores(freq_data: Dict, probabilities: Dict, weights: Dict) -> Dict[str, Dict[int, float]]:
    """
    根据所有分析结果（频率、遗漏、ML预测），使用加权公式计算每个球的最终推荐分数。

    Args:
        freq_data (Dict): 来自 `analyze_frequency_omission` 的频率和遗漏分析结果。
        probabilities (Dict): 来自机器学习模型的预测概率。
        weights (Dict): 包含所有评分权重的配置字典。

    Returns:
        Dict[str, Dict[int, float]]: 包含红球和蓝球归一化后分数的字典。
    """
    r_scores, b_scores = {}, {}
    r_freq, b_freq = freq_data.get('red_freq', {}), freq_data.get('blue_freq', {})
    omission, avg_int = freq_data.get('current_omission', {}), freq_data.get('average_interval', {})
    max_hist_o, recent_freq = freq_data.get('max_historical_omission_red', {}), freq_data.get('recent_N_freq_red', {})
    r_pred, b_pred = probabilities.get('red', {}), probabilities.get('blue', {})
    
    # 获取热冷号码分析数据
    red_hot_5 = freq_data.get('red_hot_5_periods', {})
    red_cold_6 = freq_data.get('red_cold_6_periods', {})
    blue_hot_5 = freq_data.get('blue_hot_5_periods', {})
    blue_cold_6 = freq_data.get('blue_cold_6_periods', {})
    
    # 红球评分
    for num in RED_BALL_RANGE:
        # 频率分：出现次数越多，得分越高
        freq_s = (r_freq.get(num, 0)) * weights['FREQ_SCORE_WEIGHT']
        # 遗漏分：当前遗漏接近平均遗漏时得分最高，过冷或过热都会降低分数
        omit_s = np.exp(-0.005 * (omission.get(num, 0) - avg_int.get(num, 0))**2) * weights['OMISSION_SCORE_WEIGHT']
        # 最大遗漏比率分：当前遗漏接近或超过历史最大遗漏时得分高（博冷）
        max_o_ratio = (omission.get(num, 0) / max_hist_o.get(num, 1)) if max_hist_o.get(num, 0) > 0 else 0
        max_o_s = max_o_ratio * weights['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED']
        # 近期频率分：近期出现次数越多，得分越高（追热）
        recent_s = recent_freq.get(num, 0) * weights['RECENT_FREQ_SCORE_WEIGHT_RED']
        # ML预测分
        ml_s = r_pred.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT_RED']
        # 热号加成：最近5期内出现过的号码
        hot_bonus = red_hot_5.get(num, 0) * weights['HOT_NUMBERS_5_PERIODS_WEIGHT']
        # 冷号加成：最近6期内没有出现过的号码
        cold_bonus = red_cold_6.get(num, 0) * weights['COLD_NUMBERS_6_PERIODS_WEIGHT']
        
        r_scores[num] = sum([freq_s, omit_s, max_o_s, recent_s, ml_s, hot_bonus, cold_bonus])
        
    # 蓝球评分
    blue_omission = freq_data.get('blue_current_omission', {})
    blue_avg_int = freq_data.get('blue_average_interval', {})
    blue_recent_freq = freq_data.get('recent_N_freq_blue', {})
    
    for num in BLUE_BALL_RANGE:
        freq_s = (b_freq.get(num, 0)) * weights['BLUE_FREQ_SCORE_WEIGHT']
        omit_s = np.exp(-0.01 * (blue_omission.get(num, 0) - blue_avg_int.get(num, 0))**2) * weights['BLUE_OMISSION_SCORE_WEIGHT']
        ml_s = b_pred.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT_BLUE']
        # 蓝球热号加成：最近5期内出现过的号码
        blue_hot_bonus = blue_hot_5.get(num, 0) * weights['BLUE_HOT_NUMBERS_5_PERIODS_WEIGHT']
        # 蓝球冷号加成：最近6期内没有出现过的号码
        blue_cold_bonus = blue_cold_6.get(num, 0) * weights['BLUE_COLD_NUMBERS_6_PERIODS_WEIGHT']
        
        b_scores[num] = sum([freq_s, omit_s, ml_s, blue_hot_bonus, blue_cold_bonus])

    # 归一化所有分数到0-100范围，便于比较
    def normalize_scores(scores_dict):
        if not scores_dict: return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return {k: 50.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) * 100 for k, v in scores_dict.items()}

    # 应用分布平衡调整（保持分析为基础，但鼓励分布均衡）
    normalized_r_scores = normalize_scores(r_scores)
    normalized_b_scores = normalize_scores(b_scores)
    
    # 红球区间平衡调整
    if normalized_r_scores and weights.get('ENABLE_DISTRIBUTION_BALANCE', True):
        # 计算各区间的平均分数和最高分数
        zone_stats = {1: [], 2: [], 3: []}
        for num, score in normalized_r_scores.items():
            if 1 <= num <= 12:
                zone = 1
            elif 13 <= num <= 24:
                zone = 2
            else:
                zone = 3
            zone_stats[zone].append((num, score))
        
        # 对每个区间的过高分数进行适度调整
        adjustment_factor = 0.88  # 调整强度：保持87%的原分数
        for zone, num_scores in zone_stats.items():
            if len(num_scores) > 1:
                scores = [score for _, score in num_scores]
                zone_avg = sum(scores) / len(scores)
                zone_max = max(scores)
                threshold = zone_avg + (zone_max - zone_avg) * 0.6  # 调整阈值
                
                # 只调整明显超出阈值的分数
                for num, score in num_scores:
                    if score > threshold:
                        normalized_r_scores[num] = score * adjustment_factor + zone_avg * (1 - adjustment_factor)
    
    # 蓝球大小号平衡调整
    if normalized_b_scores and weights.get('ENABLE_DISTRIBUTION_BALANCE', True):
        small_nums = [(num, score) for num, score in normalized_b_scores.items() if num <= 6]
        large_nums = [(num, score) for num, score in normalized_b_scores.items() if num > 6]
        
        adjustment_factor = 0.88
        
        for nums_group in [small_nums, large_nums]:
            if len(nums_group) > 1:
                scores = [score for _, score in nums_group]
                group_avg = sum(scores) / len(scores)
                group_max = max(scores)
                threshold = group_avg + (group_max - group_avg) * 0.6
                
                for num, score in nums_group:
                    if score > threshold:
                        normalized_b_scores[num] = score * adjustment_factor + group_avg * (1 - adjustment_factor)

    return {'red_scores': normalized_r_scores, 'blue_scores': normalized_b_scores}

def generate_combinations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """根据评分和策略生成最终的推荐组合。"""
    num_to_gen = weights_config['NUM_COMBINATIONS_TO_GENERATE']
    r_scores, b_scores = scores_data.get('red_scores', {}), scores_data.get('blue_scores', {})
    if not r_scores or not b_scores: return [], ["无法生成推荐 (分数数据缺失)。"]

    # 1. 构建候选池 - 确保号码分布的多样性
    top_n_red = int(weights_config['TOP_N_RED_FOR_CANDIDATE'])
    top_n_blue = int(weights_config['TOP_N_BLUE_FOR_CANDIDATE'])
    
    # 按分数排序，但在选择候选池时加入分区平衡
    all_red_sorted = sorted(r_scores.items(), key=lambda i: i[1], reverse=True)
    all_blue_sorted = sorted(b_scores.items(), key=lambda i: i[1], reverse=True)
    
    # 严格按分数选择候选池，确保所有号码都基于分析结果
    r_cand_pool = [num for num, score in all_red_sorted[:top_n_red]]
    b_cand_pool = [num for num, score in all_blue_sorted[:top_n_blue]]
    
    if len(r_cand_pool) < 5 or not b_cand_pool: 
        return [], ["候选池号码不足。"]

    # 2. 基于候选池的随机组合生成（候选池已基于完整分析）
    gen_pool = []
    unique_combos = set()
    
    # 创建权重用于加权随机选择
    r_weights = np.array([r_scores.get(num, 0) for num in r_cand_pool])
    r_probs = r_weights / r_weights.sum() if r_weights.sum() > 0 else None
    
    # 生成目标数量的组合
    target_combinations = max(num_to_gen * 50, 500)
    max_attempts = target_combinations * 10
    
    for attempt in range(max_attempts):
        if len(gen_pool) >= target_combinations:
            break
            
        try:
            # 从分析得出的候选池中随机选择（基于权重）
            if r_probs is not None:
                reds = sorted(np.random.choice(r_cand_pool, size=5, replace=False, p=r_probs).tolist())
            else:
                reds = sorted(random.sample(r_cand_pool, 5))
            
            # 蓝球从候选池中随机选择
            blues = sorted(random.sample(b_cand_pool, min(2, len(b_cand_pool))))
            
            # 检查是否重复
            combo_tuple = (tuple(reds), tuple(blues))
            if combo_tuple not in unique_combos:
                gen_pool.append({'red': reds, 'blue': blues})
                unique_combos.add(combo_tuple)
                
        except (ValueError, IndexError):
            # 如果候选池不足，生成其他组合
            continue

    # 3. 评分和筛选（保持原有逻辑）
    scored_combos = []
    for c in gen_pool:
        base_score = sum(r_scores.get(r, 0) for r in c['red']) + sum(b_scores.get(b, 0) for b in c['blue'])
        scored_combos.append({'combination': c, 'score': base_score, 'red_tuple': tuple(c['red'])})

    # 4. 多样性筛选和最终选择
    sorted_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True)
    final_recs = []
    max_common = 5 - int(weights_config.get('DIVERSITY_MIN_DIFFERENT_REDS', 2))
    
    if sorted_combos:
        final_recs.append(sorted_combos.pop(0))
        for cand in sorted_combos:
            if len(final_recs) >= num_to_gen: break
            # 检查与已选组合的多样性
            if all(len(set(cand['red_tuple']) & set(rec['red_tuple'])) <= max_common for rec in final_recs):
                final_recs.append(cand)
    
    # 5. 应用反向思维策略
    applied_msg = ""
    if ENABLE_FINAL_COMBO_REVERSE:
        num_to_remove = int(len(final_recs) * weights_config.get('FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT', 0))
        if 0 < num_to_remove < len(final_recs):
            removed, final_recs = final_recs[:num_to_remove], final_recs[num_to_remove:]
            applied_msg = f" (反向策略: 移除前{num_to_remove}注"
            if ENABLE_REVERSE_REFILL:
                # 补充被移除的组合
                refill_candidates = [c for c in sorted_combos if c not in final_recs and c not in removed]
                final_recs.extend(refill_candidates[:num_to_remove])
                applied_msg += "并补充)"
            else:
                applied_msg += ")"

    final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:num_to_gen]

    # 6. 生成输出字符串
    output_strs = [f"推荐组合 (Top {len(final_recs)}{applied_msg}):"]
    for i, c in enumerate(final_recs):
        r_str = ' '.join(f'{n:02d}' for n in c['combination']['red'])
        b_str = ' '.join(f'{n:02d}' for n in c['combination']['blue'])
        output_strs.append(f"  注 {i+1}: 红球 [{r_str}] 蓝球 [{b_str}] (综合分: {c['score']:.2f})")
        
    return final_recs, output_strs

def run_analysis_and_recommendation(df_hist: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame) -> Tuple:
    """
    执行一次完整的分析和推荐流程，用于特定一期。

    Returns:
        tuple: 包含推荐组合、输出字符串、分析摘要、训练模型和分数的元组。
    """
    freq_data = analyze_frequency_omission(df_hist)
    patt_data = analyze_patterns(df_hist)
    ml_models = train_prediction_models(df_hist, ml_lags)
    probabilities = predict_next_draw_probabilities(df_hist, ml_models, ml_lags) if ml_models else {'red': {}, 'blue': {}}
    scores = calculate_scores(freq_data, probabilities, weights_config)
    recs, rec_strings = generate_combinations(scores, patt_data, arm_rules, weights_config)
    analysis_summary = {'frequency_omission': freq_data, 'patterns': patt_data}
    return recs, rec_strings, analysis_summary, ml_models, scores

def run_backtest(full_df: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame, num_periods: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    在历史数据上执行策略回测，以评估策略表现。

    Returns:
        tuple: 包含详细回测结果的DataFrame和统计摘要的字典。
    """
    min_data_needed = (max(ml_lags) if ml_lags else 0) + MIN_POSITIVE_SAMPLES_FOR_ML + num_periods
    if len(full_df) < min_data_needed:
        logger.error(f"数据不足以回测{num_periods}期。需要至少{min_data_needed}期，当前有{len(full_df)}期。")
        return pd.DataFrame(), {}

    start_idx = len(full_df) - num_periods
    results, prize_counts = [], Counter()
    red_cols = [f'红球_{i}' for i in range(1, 6)]  # 大乐透5个红球
    blue_cols = [f'蓝球_{i}' for i in range(1, 3)]  # 大乐透2个蓝球
    best_hits_per_period = []
    
    logger.info("策略回测已启动...")
    start_time = time.time()
    
    for i in range(num_periods):
        current_iter = i + 1
        current_idx = start_idx + i
        
        # 使用SuppressOutput避免在回测循环中打印大量日志
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
            # 临时设置更高的日志级别以减少输出
            temp_level = logger.level
            logger.setLevel(logging.WARNING)
            hist_data = full_df.iloc[:current_idx]
            predicted_combos, _, _, _, _ = run_analysis_and_recommendation(hist_data, ml_lags, weights_config, arm_rules)
            logger.setLevel(temp_level)
            
        actual_outcome = full_df.loc[current_idx]
        actual_red_set = set(actual_outcome[red_cols])
        actual_blue_set = set(actual_outcome[blue_cols])
        
        period_max_red_hits, period_blue_hits_on_max_red = 0, 0
        if not predicted_combos:
            best_hits_per_period.append({'period': actual_outcome['期号'], 'best_red_hits': 0, 'blue_hits': 0, 'prize': None})
        else:
            for combo_dict in predicted_combos:
                combo = combo_dict['combination']
                red_hits = len(set(combo['red']) & actual_red_set)
                blue_hits = len(set(combo['blue']) & actual_blue_set)
                prize = get_prize_level(red_hits, blue_hits)
                if prize: prize_counts[prize] += 1
                results.append({'period': actual_outcome['期号'], 'red_hits': red_hits, 'blue_hits': blue_hits, 'prize': prize})
                
                if red_hits > period_max_red_hits:
                    period_max_red_hits, period_blue_hits_on_max_red = red_hits, blue_hits
                elif red_hits == period_max_red_hits and blue_hits > period_blue_hits_on_max_red:
                    period_blue_hits_on_max_red = blue_hits
            
            best_hits_per_period.append({
                'period': actual_outcome['期号'], 
                'best_red_hits': period_max_red_hits, 
                'blue_hits': period_blue_hits_on_max_red, 
                'prize': get_prize_level(period_max_red_hits, period_blue_hits_on_max_red)
            })

        # 打印进度
        if current_iter == 1 or current_iter % 10 == 0 or current_iter == num_periods:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_iter
            remaining_time = avg_time * (num_periods - current_iter)
            progress_logger.info(f"回测进度: {current_iter}/{num_periods} | 平均耗时: {avg_time:.2f}s/期 | 预估剩余: {format_time(remaining_time)}")
            
    return pd.DataFrame(results), {'prize_counts': dict(prize_counts), 'best_hits_per_period': pd.DataFrame(best_hits_per_period)}

def objective(trial: optuna.trial.Trial, df_for_opt: pd.DataFrame, ml_lags: List[int], arm_rules: pd.DataFrame) -> float:
    """Optuna 的目标函数，用于评估一组给定的权重参数的好坏。"""
    trial_weights = {}
    
    # 动态地从DEFAULT_WEIGHTS构建搜索空间
    for key, value in DEFAULT_WEIGHTS.items():
        if isinstance(value, int):
            if 'NUM_COMBINATIONS' in key: trial_weights[key] = trial.suggest_int(key, 5, 15)
            elif 'TOP_N' in key: trial_weights[key] = trial.suggest_int(key, 15, 25)  # 大乐透调整
            else: trial_weights[key] = trial.suggest_int(key, max(0, value - 2), value + 2)
        elif isinstance(value, float):
            # 对不同类型的浮点数使用不同的搜索范围
            if any(k in key for k in ['PERCENT', 'FACTOR', 'SUPPORT', 'CONFIDENCE']):
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 1.5)
            else: # 对权重参数使用更宽的搜索范围
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 2.0)

    full_trial_weights = DEFAULT_WEIGHTS.copy()
    full_trial_weights.update(trial_weights)
    
    # 在快速回测中评估这组权重
    with SuppressOutput():
        _, backtest_stats = run_backtest(df_for_opt, ml_lags, full_trial_weights, arm_rules, OPTIMIZATION_BACKTEST_PERIODS)
        
    # 定义一个分数来衡量表现，高奖金等级的权重更高
    prize_weights = {'一等奖': 10000, '二等奖': 2000, '三等奖': 500, '四等奖': 100, '五等奖': 20, '六等奖': 10, '七等奖': 5, '八等奖': 2, '九等奖': 1}
    score = sum(prize_weights.get(p, 0) * c for p, c in backtest_stats.get('prize_counts', {}).items())
    return score

def optuna_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, total_trials: int):
    """Optuna 的回调函数，用于在控制台报告优化进度。"""
    global OPTUNA_START_TIME
    current_iter = trial.number + 1
    if current_iter == 1 or current_iter % 10 == 0 or current_iter == total_trials:
        elapsed = time.time() - OPTUNA_START_TIME
        avg_time = elapsed / current_iter
        remaining_time = avg_time * (total_trials - current_iter)
        best_value = f"{study.best_value:.2f}" if study.best_trial else "N/A"
        progress_logger.info(f"Optuna进度: {current_iter}/{total_trials} | 当前最佳得分: {best_value} | 预估剩余: {format_time(remaining_time)}")

if __name__ == "__main__":
    # 1. 初始化日志记录器，同时输出到控制台和文件
    log_filename = os.path.join(SCRIPT_DIR, f"dlt_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    set_console_verbosity(logging.INFO, use_simple_formatter=True)

    logger.info("--- 大乐透数据分析与推荐系统 ---")
    logger.info("启动数据加载和预处理...")

    # 2. 健壮的数据加载逻辑
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH):
        main_df = load_data(PROCESSED_CSV_PATH)
        if main_df is not None:
             logger.info("从缓存文件加载预处理数据成功。")

    if main_df is None or main_df.empty:
        logger.info("未找到或无法加载缓存数据，正在从原始文件生成...")
        raw_df = load_data(CSV_FILE_PATH)
        if raw_df is not None and not raw_df.empty:
            logger.info("原始数据加载成功，开始清洗...")
            cleaned_df = clean_and_structure(raw_df)
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("数据清洗成功，开始特征工程...")
                main_df = feature_engineer(cleaned_df)
                if main_df is not None and not main_df.empty:
                    logger.info("特征工程成功，保存预处理数据...")
                    try:
                        main_df.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"预处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except IOError as e:
                        logger.error(f"保存预处理数据失败: {e}")
                else:
                    logger.error("特征工程失败，无法生成最终数据集。")
            else:
                logger.error("数据清洗失败。")
        else:
            logger.error("原始数据加载失败。")
    
    if main_df is None or main_df.empty:
        logger.critical("数据准备失败，无法继续。请检查 'dlt_data_processor.py' 是否已成功运行并生成 'daletou.csv'。程序终止。")
        sys.exit(1)
    
    logger.info(f"数据加载完成，共 {len(main_df)} 期有效数据。")
    last_period = main_df['期号'].iloc[-1]

    # 3. 根据模式执行：优化或直接分析
    active_weights = DEFAULT_WEIGHTS.copy()
    optuna_summary = None

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info("\n" + "="*25 + " Optuna 参数优化模式 " + "="*25)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        # 优化前先进行一次全局关联规则分析
        optuna_arm_rules = analyze_associations(main_df, DEFAULT_WEIGHTS)
        
        study = optuna.create_study(direction="maximize")
        global OPTUNA_START_TIME; OPTUNA_START_TIME = time.time()
        progress_callback_with_total = partial(optuna_progress_callback, total_trials=OPTIMIZATION_TRIALS)
        
        try:
            study.optimize(lambda t: objective(t, main_df, ML_LAG_FEATURES, optuna_arm_rules), n_trials=OPTIMIZATION_TRIALS, callbacks=[progress_callback_with_total])
            logger.info("Optuna 优化完成。")
            active_weights.update(study.best_params)
            optuna_summary = {"status": "完成", "best_value": study.best_value, "best_params": study.best_params}
        except Exception as e:
            logger.error(f"Optuna 优化过程中断: {e}", exc_info=True)
            optuna_summary = {"status": "中断", "error": str(e)}
            logger.warning("优化中断，将使用默认权重继续分析。")
    
    # 4. 切换到报告模式并打印报告头
    report_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(report_formatter)
    global_console_handler.setFormatter(report_formatter)
    
    logger.info("\n\n" + "="*60 + f"\n{' ' * 18}大乐透策略分析报告\n" + "="*60)
    logger.info(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"分析基于数据: 截至 {last_period} 期 (共 {len(main_df)} 期)")
    logger.info(f"本次预测目标: 第 {last_period + 1} 期")
    logger.info(f"日志文件: {os.path.basename(log_filename)}")

    # 5. 打印优化摘要
    if ENABLE_OPTUNA_OPTIMIZATION and optuna_summary:
        logger.info("\n" + "="*25 + " Optuna 优化摘要 " + "="*25)
        logger.info(f"优化状态: {optuna_summary['status']}")
        if optuna_summary['status'] == '完成':
            logger.info(f"最佳性能得分: {optuna_summary['best_value']:.4f}")
            logger.info("--- 本次分析已采用以下优化参数 ---")
            best_params_str = json.dumps(optuna_summary['best_params'], indent=2, ensure_ascii=False)
            logger.info(best_params_str)
        else: logger.info(f"错误信息: {optuna_summary['error']}")
    else:
        logger.info("\n--- 本次分析使用脚本内置的默认权重 ---")

    # 6. 全局分析
    full_history_arm_rules = analyze_associations(main_df, active_weights)
    
    # 7. 回测并打印报告
    logger.info("\n" + "="*25 + " 策 略 回 测 摘 要 " + "="*25)
    backtest_results_df, backtest_stats = run_backtest(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
    
    if not backtest_results_df.empty:
        num_periods_tested = len(backtest_results_df['period'].unique())
        num_combos_per_period = active_weights.get('NUM_COMBINATIONS_TO_GENERATE', 10)
        total_bets = len(backtest_results_df)
        logger.info(f"回测周期: 最近 {num_periods_tested} 期 | 每期注数: {num_combos_per_period} | 总投入注数: {total_bets}")
        logger.info("\n--- 1. 奖金与回报分析 ---")
        prize_dist = backtest_stats.get('prize_counts', {})
        # 大乐透奖金设置 (基于2024年奖金水平估算)
        prize_values = {
            '一等奖': 10000000, '二等奖': 800000, '三等奖': 10000, 
            '四等奖': 3000, '五等奖': 300, '六等奖': 200, 
            '七等奖': 100, '八等奖': 15, '九等奖': 5
        }
        total_revenue = sum(prize_values.get(p, 0) * c for p, c in prize_dist.items())
        total_cost = total_bets * 3  # 大乐透单注3元
        roi = (total_revenue - total_cost) * 100 / total_cost if total_cost > 0 else 0
        logger.info(f"  - 估算总回报: {total_revenue:,.2f} 元 (总成本: {total_cost:,.2f} 元)")
        logger.info(f"  - 投资回报率 (ROI): {roi:.2f}%")
        logger.info("  - 中奖等级分布 (总计):")
        if prize_dist:
            for prize in prize_values.keys():
                if prize in prize_dist: logger.info(f"    - {prize:<4s}: {prize_dist[prize]:>4d} 次")
        else: logger.info("    - 未命中任何奖级。")
        logger.info("\n--- 2. 核心性能指标 ---")
        logger.info(f"  - 平均红球命中 (每注): {backtest_results_df['red_hits'].mean():.3f} / 5")
        logger.info(f"  - 平均蓝球命中 (每注): {backtest_results_df['blue_hits'].mean():.3f} / 2")
        logger.info("\n--- 3. 每期最佳命中表现 ---")
        if (best_hits_df := backtest_stats.get('best_hits_per_period')) is not None and not best_hits_df.empty:
            logger.info("  - 在一期内至少命中:")
            # 大乐透奖级统计
            for prize_name, prize_query in [
                ("四等奖(4+2)", "`best_red_hits` == 4 and `blue_hits` == 2"),
                ("五等奖(4+1或3+2)", "(`best_red_hits` == 4 and `blue_hits` == 1) or (`best_red_hits` == 3 and `blue_hits` == 2)"),
                ("三等奖(5+0)", "`best_red_hits` == 5 and `blue_hits` == 0"),
                ("二等奖(5+1)", "`best_red_hits` == 5 and `blue_hits` == 1"),
                ("一等奖(5+2)", "`best_red_hits` == 5 and `blue_hits` == 2")
            ]:
                count = best_hits_df.query(prize_query).shape[0] if not best_hits_df.empty else 0
                logger.info(f"    - {prize_name:<15s}: {count} / {num_periods_tested} 期")
            any_blue_hit_periods = (best_hits_df['blue_hits'] > 0).sum()
            logger.info(f"  - 蓝球覆盖率: 在 {any_blue_hit_periods / num_periods_tested:.2%} 的期数中，推荐组合至少命中1个蓝球")
    else: logger.warning("回测未产生有效结果，可能是数据量不足。")
    
    # 8. 最终推荐
    logger.info("\n" + "="*25 + f" 第 {last_period + 1} 期 号 码 推 荐 " + "="*25)
    final_recs, final_rec_strings, _, _, final_scores = run_analysis_and_recommendation(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules)
    
    logger.info("\n--- 单式推荐 ---")
    for line in final_rec_strings: logger.info(line)
    
    logger.info("\n--- 复式参考 ---")
    if final_scores and final_scores.get('red_scores'):
        # 采用分区平衡的复式选择策略，避免大号偏向
        r_scores = final_scores['red_scores']
        b_scores = final_scores['blue_scores']
        
        # 显示热冷号码分析
        logger.info("\n--- 热冷号码分析 (基于经验规律) ---")
        freq_analysis = analyze_frequency_omission(main_df)
        
        # 红球热号（最近5期内出现过）
        red_hot_nums = [num for num in RED_BALL_RANGE if freq_analysis.get('red_hot_5_periods', {}).get(num, 0) == 1]
        # 红球冷号（最近6期内未出现）
        red_cold_nums = [num for num in RED_BALL_RANGE if freq_analysis.get('red_cold_6_periods', {}).get(num, 0) == 1]
        # 蓝球热号
        blue_hot_nums = [num for num in BLUE_BALL_RANGE if freq_analysis.get('blue_hot_5_periods', {}).get(num, 0) == 1]
        # 蓝球冷号
        blue_cold_nums = [num for num in BLUE_BALL_RANGE if freq_analysis.get('blue_cold_6_periods', {}).get(num, 0) == 1]
        
        logger.info(f"  红球热号(最近5期): {' '.join(f'{n:02d}' for n in sorted(red_hot_nums))} (共{len(red_hot_nums)}个)")
        logger.info(f"  红球冷号(6期未出): {' '.join(f'{n:02d}' for n in sorted(red_cold_nums))} (共{len(red_cold_nums)}个)")
        logger.info(f"  蓝球热号(最近5期): {' '.join(f'{n:02d}' for n in sorted(blue_hot_nums))} (共{len(blue_hot_nums)}个)")
        logger.info(f"  蓝球冷号(6期未出): {' '.join(f'{n:02d}' for n in sorted(blue_cold_nums))} (共{len(blue_cold_nums)}个)")
        logger.info(f"  策略权重: 红球热号{DEFAULT_WEIGHTS['HOT_NUMBERS_5_PERIODS_WEIGHT']} + 冷号{DEFAULT_WEIGHTS['COLD_NUMBERS_6_PERIODS_WEIGHT']}")
        logger.info(f"  策略权重: 蓝球热号{DEFAULT_WEIGHTS['BLUE_HOT_NUMBERS_5_PERIODS_WEIGHT']} + 冷号{DEFAULT_WEIGHTS['BLUE_COLD_NUMBERS_6_PERIODS_WEIGHT']}")
        
        # 红球分区选择：确保各区间平衡
        zone_reds = {1: [], 2: [], 3: []}
        for num, score in sorted(r_scores.items(), key=lambda x: x[1], reverse=True):
            if 1 <= num <= 12:
                zone_reds[1].append(num)
            elif 13 <= num <= 24:
                zone_reds[2].append(num)
            else:
                zone_reds[3].append(num)
        
        # 从每个区间选择代表性号码
        complex_red = []
        target_per_zone = 8 // 3  # 每区目标数量
        remaining = 8
        
        for zone in [1, 2, 3]:
            zone_count = min(len(zone_reds[zone]), max(2, min(target_per_zone, remaining-2)))
            complex_red.extend(zone_reds[zone][:zone_count])
            remaining -= zone_count
        
        # 如果还需要更多号码，按分数补充
        all_red_by_score = [n for n, _ in sorted(r_scores.items(), key=lambda x: x[1], reverse=True)]
        for num in all_red_by_score:
            if num not in complex_red and len(complex_red) < 8:
                complex_red.append(num)
        
        complex_red = sorted(complex_red[:8])
        
        # 蓝球大小号平衡选择
        small_blues = [n for n, s in sorted(b_scores.items(), key=lambda x: x[1], reverse=True) if n <= 6]
        large_blues = [n for n, s in sorted(b_scores.items(), key=lambda x: x[1], reverse=True) if n > 6]
        
        complex_blue = []
        complex_blue.extend(small_blues[:3])  # 最多选3个小号
        complex_blue.extend(large_blues[:3])  # 最多选3个大号
        complex_blue = sorted(complex_blue[:6])
        
        logger.info(f"  红球 (平衡选择): {' '.join(f'{n:02d}' for n in complex_red)}")
        logger.info(f"  蓝球 (平衡选择): {' '.join(f'{n:02d}' for n in complex_blue)}")
        logger.info(f"  红球分布 - 小区(1-12):{len([n for n in complex_red if 1<=n<=12])}个, 中区(13-24):{len([n for n in complex_red if 13<=n<=24])}个, 大区(25-35):{len([n for n in complex_red if 25<=n<=35])}个")
        logger.info(f"  蓝球分布 - 小号(1-6):{len([n for n in complex_blue if n<=6])}个, 大号(7-12):{len([n for n in complex_blue if n>6])}个")
    
    logger.info("\n" + "="*60 + f"\n--- 报告结束 (详情请查阅: {os.path.basename(log_filename)}) ---\n")
    
    # 发送微信推送
    try:
        from wxPusher import send_analysis_report
        logger.info("正在发送微信推送...")
        
        # 提取复式推荐号码
        complex_red_list = None
        complex_blue_list = None
        if final_scores and final_scores.get('red_scores'):
            # 重新提取复式号码（与上面的逻辑保持一致）
            r_scores = final_scores['red_scores']
            b_scores = final_scores['blue_scores']
            
            # 红球分区选择：确保各区间平衡
            zone_reds = {1: [], 2: [], 3: []}
            for num, score in sorted(r_scores.items(), key=lambda x: x[1], reverse=True):
                if 1 <= num <= 12:
                    zone_reds[1].append(num)
                elif 13 <= num <= 24:
                    zone_reds[2].append(num)
                else:
                    zone_reds[3].append(num)
            
            # 从每个区间选择代表性号码
            complex_red = []
            target_per_zone = 8 // 3  # 每区目标数量
            remaining = 8
            
            for zone in [1, 2, 3]:
                zone_count = min(len(zone_reds[zone]), max(2, min(target_per_zone, remaining-2)))
                complex_red.extend(zone_reds[zone][:zone_count])
                remaining -= zone_count
            
            # 如果还需要更多号码，按分数补充
            all_red_by_score = [n for n, _ in sorted(r_scores.items(), key=lambda x: x[1], reverse=True)]
            for num in all_red_by_score:
                if num not in complex_red and len(complex_red) < 8:
                    complex_red.append(num)
            
            complex_red = sorted(complex_red[:8])
            
            # 蓝球大小号平衡选择
            small_blues = [n for n, s in sorted(b_scores.items(), key=lambda x: x[1], reverse=True) if n <= 6]
            large_blues = [n for n, s in sorted(b_scores.items(), key=lambda x: x[1], reverse=True) if n > 6]
            
            complex_blue = []
            complex_blue.extend(small_blues[:3])  # 最多选3个小号
            complex_blue.extend(large_blues[:3])  # 最多选3个大号
            complex_blue = sorted(complex_blue[:6])
            
            # 转换为格式化字符串列表
            complex_red_list = [f'{n:02d}' for n in complex_red]
            complex_blue_list = [f'{n:02d}' for n in complex_blue]
        
        # 发送分析报告推送
        push_result = send_analysis_report(
            report_content=log_filename,  # 报告文件名
            period=last_period + 1,      # 预测期号
            recommendations=final_rec_strings,  # 推荐号码
            complex_red=complex_red_list,       # 复式红球
            complex_blue=complex_blue_list      # 复式蓝球
        )
        
        if push_result.get("success", False):
            logger.info("✅ 微信推送发送成功")
        else:
            logger.warning(f"⚠️ 微信推送发送失败: {push_result.get('error', '未知错误')}")
            
    except ImportError:
        logger.warning("微信推送模块未找到，跳过推送功能")
    except Exception as e:
        logger.error(f"微信推送发送异常: {e}") 