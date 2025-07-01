# -*- coding: utf-8 -*-
"""
大乐透推荐结果验证与奖金计算器
=================================

本脚本旨在自动评估 `dlt_analyzer.py` 生成的推荐号码的实际表现。

工作流程:
1.  读取 `daletou.csv` 文件，获取所有历史开奖数据。
2.  确定最新的一期为"评估期"，倒数第二期为"报告数据截止期"。
3.  根据"报告数据截止期"，在当前目录下查找对应的分析报告文件
    (dlt_analysis_output_*.txt)。
4.  从找到的报告中解析出"单式推荐"和"复式参考"的号码。
5.  将复式参考号码展开为所有可能的单式投注。
6.  使用"评估期"的实际开奖号码，核对所有推荐投注的中奖情况。
7.  计算总奖金，并将详细的中奖结果（包括中奖号码、奖级、金额）
    追加记录到主报告文件 `latest_dlt_calculation.txt` 中。
8.  主报告文件会自动管理记录数量，只保留最新的N条评估记录和错误日志。
"""

import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime
import traceback
from typing import Optional, Tuple, List, Dict

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 脚本需要查找的分析报告文件名的模式
REPORT_PATTERN = "dlt_analysis_output_*.txt"
# 开奖数据源CSV文件
CSV_FILE = "daletou.csv"
# 最终生成的主评估报告文件名
MAIN_REPORT_FILE = "latest_dlt_calculation.txt"

# 主报告文件中保留的最大记录数
MAX_NORMAL_RECORDS = 10  # 保留最近10次评估
MAX_ERROR_LOGS = 20      # 保留最近20条错误日志

# 大乐透奖金对照表 (元) - 官方标准奖金设置
PRIZE_TABLE = {
    1: 10_000_000,  # 一等奖 (1000万元，浮动)
    2: 300_000,     # 二等奖 (30万元，浮动)
    3: 10_000,      # 三等奖 (1万元，固定)
    4: 3_000,       # 四等奖 (3000元，固定)
    5: 300,         # 五等奖 (300元，固定)
    6: 200,         # 六等奖 (200元，固定)
    7: 100,         # 七等奖 (100元，固定)
    8: 15,          # 八等奖 (15元，固定)
    9: 5,           # 九等奖 (5元，固定)
}

# 大乐透每注价格
TICKET_PRICE = 2  # 每注2元

# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

def log_message(message: str, level: str = "INFO"):
    """一个简单的日志打印函数，用于在控制台显示脚本执行状态。"""
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')} - {message}")

def robust_file_read(file_path: str) -> Optional[str]:
    """
    一个健壮的文件读取函数，能自动尝试多种编码格式。

    Args:
        file_path (str): 待读取文件的路径。

    Returns:
        Optional[str]: 文件内容字符串，如果失败则返回 None。
    """
    if not os.path.exists(file_path):
        log_message(f"文件未找到: {file_path}", "ERROR")
        return None
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            continue
    log_message(f"无法使用任何支持的编码打开文件: {file_path}", "ERROR")
    return None

# ==============================================================================
# --- 数据解析与查找模块 ---
# ==============================================================================

def get_period_data_from_csv(csv_content: str) -> Tuple[Optional[Dict], Optional[List]]:
    """
    从CSV文件内容中解析出所有期号的开奖数据。

    Args:
        csv_content (str): 从CSV文件读取的字符串内容。

    Returns:
        Tuple[Optional[Dict], Optional[List]]:
            - 一个以期号为键，开奖数据为值的字典。
            - 一个按升序排序的期号列表。
            如果解析失败则返回 (None, None)。
    """
    if not csv_content:
        log_message("输入的CSV内容为空。", "WARNING")
        return None, None
    period_map, periods_list = {}, []
    try:
        reader = csv.reader(csv_content.splitlines())
        next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if len(row) >= 4 and re.match(r'^\d{4,7}$', row[0]):
                try:
                    period, date, red_str, blue_str = row[0], row[1], row[2], row[3]
                    
                    # 跳过没有日期的记录（通常是重复数据）
                    if not date or date.strip() == "":
                        continue
                    
                    # 大乐透：5个红球，2个蓝球
                    red_balls = sorted(map(int, red_str.split(',')))
                    blue_balls = sorted(map(int, blue_str.split(',')))
                    
                    if (len(red_balls) != 5 or len(blue_balls) != 2 or 
                        not all(1 <= r <= 35 for r in red_balls) or 
                        not all(1 <= b <= 12 for b in blue_balls)):
                        continue
                    
                    # 标准化期号格式（使用简化格式存储）
                    normalized_period = normalize_period(period)
                    
                    # 避免重复：如果已存在该期号，保留有更多信息的记录
                    if normalized_period not in period_map or date:
                        period_map[normalized_period] = {'date': date, 'red': red_balls, 'blue': blue_balls}
                        if normalized_period not in periods_list:
                            periods_list.append(normalized_period)
                            
                except (ValueError, IndexError):
                    log_message(f"CSV文件第 {i+2} 行数据格式无效，已跳过: {row}", "WARNING")
    except Exception as e:
        log_message(f"解析CSV数据时发生严重错误: {e}", "ERROR")
        return None, None
    
    if not period_map:
        log_message("未能从CSV中解析到任何有效的开奖数据。", "WARNING")
        return None, None
        
    # 按期号排序（简化格式的数字排序）
    periods_list = sorted(list(set(periods_list)), key=lambda x: int(x))
    log_message(f"成功解析 {len(period_map)} 期有效数据，最新期号: {periods_list[-1] if periods_list else 'N/A'}")
    
    return period_map, periods_list

def normalize_period(period: str) -> str:
    """
    标准化期号格式：将完整格式(2025073)转换为简化格式(25073)
    
    Args:
        period (str): 原始期号
        
    Returns:
        str: 标准化后的期号（简化格式）
    """
    if len(period) == 7 and period.startswith('2025'):
        # 完整格式转简化格式：2025073 -> 25073
        return period[2:]
    elif len(period) == 5 and period.startswith('25'):
        # 已经是简化格式
        return period
    else:
        # 其他格式保持不变
        return period

def find_matching_report(target_prediction_period: str) -> Optional[str]:
    """
    在当前目录查找预测目标期号为 `target_prediction_period` 的最新分析报告。

    Args:
        target_prediction_period (str): 目标预测期号。

    Returns:
        Optional[str]: 找到的报告文件的路径，如果未找到则返回 None。
    """
    # 标准化目标期号格式
    normalized_target = normalize_period(target_prediction_period)
    log_message(f"正在查找预测期号为 {target_prediction_period}(标准化后: {normalized_target}) 的分析报告...")
    
    candidates = []
    # 使用 SCRIPT_DIR 确保在任何工作目录下都能找到与脚本同级的报告文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(os.path.join(script_dir, REPORT_PATTERN)):
        content = robust_file_read(file_path)
        if not content: continue
        
        # 修改匹配逻辑：查找"本次预测目标: 第 XXX 期"和"第 XXX 期 号 码 推 荐"
        target_match = re.search(r'本次预测目标:\s*第\s*(\d+)\s*期', content)
        recommend_match = re.search(r'第\s*(\d+)\s*期\s*号\s*码\s*推\s*荐', content)
        
        found_match = False
        period_source = ""
        
        # 优先检查推荐标题中的期号
        if recommend_match:
            report_period = normalize_period(recommend_match.group(1))
            if report_period == normalized_target:
                found_match = True
                period_source = f"推荐标题期号: {recommend_match.group(1)} -> {report_period}"
        
        # 如果推荐标题没匹配，再检查预测目标
        if not found_match and target_match:
            report_period = normalize_period(target_match.group(1))
            if report_period == normalized_target:
                found_match = True
                period_source = f"预测目标期号: {target_match.group(1)} -> {report_period}"
        
        if found_match:
            try:
                # 从文件名中提取时间戳以确定最新报告
                timestamp_str_match = re.search(r'_(\d{8}_\d{6})\.txt$', file_path)
                if timestamp_str_match:
                    timestamp_str = timestamp_str_match.group(1)
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((timestamp, file_path))
                    log_message(f"找到匹配报告: {os.path.basename(file_path)} ({period_source})")
            except (AttributeError, ValueError):
                continue
    
    if not candidates:
        log_message(f"未找到预测期号为 {normalized_target} 的分析报告。", "WARNING")
        return None
        
    candidates.sort(reverse=True)
    latest_report = candidates[0][1]
    log_message(f"找到匹配的最新报告: {os.path.basename(latest_report)}", "INFO")
    return latest_report

def parse_recommendations_from_report(content: str) -> Tuple[List, List, List]:
    """
    从分析报告内容中解析出单式和复式推荐号码。

    Args:
        content (str): 分析报告的文本内容。

    Returns:
        Tuple[List, List, List]:
            - 单式推荐列表, e.g., [([1,2,3,4,5], [6,7]), ...]
            - 复式红球列表, e.g., [1,2,3,4,5,6,7]
            - 复式蓝球列表, e.g., [8,9,10]
    """
    # 解析单式推荐（大乐透格式）
    rec_pattern = re.compile(r'注\s*\d+:\s*红球\s*\[([\d\s]+)\]\s*蓝球\s*\[([\d\s]+)\]')
    rec_tickets = []
    for match in rec_pattern.finditer(content):
        try:
            reds = sorted(map(int, match.group(1).split()))
            blues = sorted(map(int, match.group(2).split()))
            if len(reds) == 5 and len(blues) == 2: 
                rec_tickets.append((reds, blues))
        except ValueError: 
            continue
    
    # 解析复式推荐 - 支持多种格式
    complex_reds, complex_blues = [], []
    
    # 格式1: "红球 (Top X): ..."
    red_match = re.search(r'红球\s*\(Top\s*\d+\):\s*([\d\s]+)', content)
    if red_match:
        try: 
            complex_reds = sorted(map(int, red_match.group(1).split()))
        except ValueError: 
            pass
    
    # 格式2: "红球 (平衡选择): ..."
    if not complex_reds:
        red_match = re.search(r'红球\s*\(平衡选择\):\s*([\d\s]+)', content)
        if red_match:
            try: 
                complex_reds = sorted(map(int, red_match.group(1).split()))
            except ValueError: 
                pass
    
    # 格式3: "红球推荐: ..." 或其他可能的格式
    if not complex_reds:
        red_match = re.search(r'红球[推荐选择]*\s*[:：]\s*((?:\d{2}\s*)+)', content)
        if red_match:
            try: 
                complex_reds = sorted(map(int, red_match.group(1).split()))
            except ValueError: 
                pass
    
    # 蓝球复式解析 - 支持多种格式
    # 格式1: "蓝球 (Top X): ..."
    blue_match = re.search(r'蓝球\s*\(Top\s*\d+\):\s*([\d\s]+)', content)
    if blue_match:
        try: 
            complex_blues = sorted(map(int, blue_match.group(1).split()))
        except ValueError: 
            pass
    
    # 格式2: "蓝球 (平衡选择): ..."
    if not complex_blues:
        blue_match = re.search(r'蓝球\s*\(平衡选择\):\s*([\d\s]+)', content)
        if blue_match:
            try: 
                complex_blues = sorted(map(int, blue_match.group(1).split()))
            except ValueError: 
                pass
    
    # 格式3: "蓝球推荐: ..." 或其他可能的格式
    if not complex_blues:
        blue_match = re.search(r'蓝球[推荐选择]*\s*[:：]\s*((?:\d{2}\s*)+)', content)
        if blue_match:
            try: 
                complex_blues = sorted(map(int, blue_match.group(1).split()))
            except ValueError: 
                pass
    
    return rec_tickets, complex_reds, complex_blues

def generate_complex_tickets(reds: List, blues: List) -> List:
    """
    根据复式红球和蓝球列表，生成所有可能的单式投注组合。
    大乐透：从红球中选5个，从蓝球中选2个

    Args:
        reds (List): 复式红球号码列表
        blues (List): 复式蓝球号码列表

    Returns:
        List: 所有可能的单式投注组合列表，每个元素为 (red_combination, blue_combination)
    """
    if len(reds) < 5 or len(blues) < 2:
        log_message(f"复式号码数量不足：红球需至少5个(当前{len(reds)})，蓝球需至少2个(当前{len(blues)})", "WARNING")
        return []
    
    # 检查组合数量，避免生成过多组合
    def combinations_count(n, k):
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    red_combos_count = combinations_count(len(reds), 5)
    blue_combos_count = combinations_count(len(blues), 2)
    total_combos = red_combos_count * blue_combos_count
    
    if total_combos > 10000:  # 限制最大组合数
        log_message(f"复式组合数量过大({total_combos})，可能导致计算时间过长。建议减少复式号码数量。", "WARNING")
        return []
    
    complex_tickets = []
    for red_combo in combinations(reds, 5):
        for blue_combo in combinations(blues, 2):
            complex_tickets.append((sorted(red_combo), sorted(blue_combo)))
    
    return complex_tickets

def calculate_prize_detailed(single_tickets: List, complex_tickets: List, complex_info: Optional[Dict], 
                           prize_red: List, prize_blue: List) -> Tuple[int, Dict, List, Dict]:
    """
    分别计算单式和复式投注的中奖情况和总奖金。

    Args:
        single_tickets (List): 单式投注号码列表
        complex_tickets (List): 复式展开的投注号码列表
        complex_info (Optional[Dict]): 复式投注信息
        prize_red (List): 开奖红球号码列表
        prize_blue (List): 开奖蓝球号码列表

    Returns:
        Tuple[int, Dict, List, Dict]:
            - 总奖金数额
            - 各奖级的中奖次数字典
            - 中奖投注的详细信息列表
            - 详细统计信息字典
    """
    # 计算单式投注的中奖情况
    single_prize, single_counts, single_details = calculate_prize(single_tickets, prize_red, prize_blue)
    
    # 计算复式投注的中奖情况
    complex_prize, complex_counts, complex_details = calculate_prize(complex_tickets, prize_red, prize_blue)
    
    # 合并统计信息
    total_prize = single_prize + complex_prize
    total_counts = {}
    
    # 合并中奖次数统计
    for level in set(list(single_counts.keys()) + list(complex_counts.keys())):
        total_counts[level] = single_counts.get(level, 0) + complex_counts.get(level, 0)
    
    # 合并中奖详情
    all_details = single_details + complex_details
    
    # 计算投资成本和回报率
    single_cost = len(single_tickets) * TICKET_PRICE
    complex_cost = len(complex_tickets) * TICKET_PRICE
    total_cost = single_cost + complex_cost
    
    # 计算投资回报率 (ROI)
    single_roi = ((single_prize - single_cost) / single_cost * 100) if single_cost > 0 else 0
    complex_roi = ((complex_prize - complex_cost) / complex_cost * 100) if complex_cost > 0 else 0
    total_roi = ((total_prize - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # 生成详细统计信息
    detailed_stats = {
        'single': {
            'tickets_count': len(single_tickets),
            'cost_amount': single_cost,
            'prize_amount': single_prize,
            'net_profit': single_prize - single_cost,
            'roi_percent': single_roi,
            'prize_counts': single_counts,
            'winning_details': single_details
        },
        'complex': {
            'tickets_count': len(complex_tickets),
            'cost_amount': complex_cost,
            'prize_amount': complex_prize,
            'net_profit': complex_prize - complex_cost,
            'roi_percent': complex_roi,
            'prize_counts': complex_counts,
            'winning_details': complex_details,
            'info': complex_info
        },
        'total': {
            'tickets_count': len(single_tickets) + len(complex_tickets),
            'cost_amount': total_cost,
            'prize_amount': total_prize,
            'net_profit': total_prize - total_cost,
            'roi_percent': total_roi,
            'prize_counts': total_counts
        }
    }
    
    return total_prize, total_counts, all_details, detailed_stats

def calculate_prize(tickets: List, prize_red: List, prize_blue: List) -> Tuple[int, Dict, List]:
    """
    计算投注列表的中奖情况和总奖金。

    Args:
        tickets (List): 投注号码列表，每个元素为 (red_list, blue_list)
        prize_red (List): 开奖红球号码列表
        prize_blue (List): 开奖蓝球号码列表

    Returns:
        Tuple[int, Dict, List]:
            - 总奖金数额
            - 各奖级的中奖次数字典
            - 中奖投注的详细信息列表
    """
    total_prize = 0
    prize_counts = {}
    winning_details = []
    
    for ticket in tickets:
        red_ticket, blue_ticket = ticket
        
        # 计算命中数
        red_hits = len(set(red_ticket) & set(prize_red))
        blue_hits = len(set(blue_ticket) & set(prize_blue))
        
        # 根据大乐透官方规则确定奖级
        prize_level = None
        if red_hits == 5 and blue_hits == 2:
            prize_level = 1  # 一等奖
        elif red_hits == 5 and blue_hits == 1:
            prize_level = 2  # 二等奖
        elif red_hits == 5 and blue_hits == 0:
            prize_level = 3  # 三等奖
        elif red_hits == 4 and blue_hits == 2:
            prize_level = 4  # 四等奖
        elif red_hits == 4 and blue_hits == 1:
            prize_level = 5  # 五等奖
        elif red_hits == 3 and blue_hits == 2:
            prize_level = 6  # 六等奖
        elif red_hits == 4 and blue_hits == 0:
            prize_level = 7  # 七等奖
        elif (red_hits == 3 and blue_hits == 1) or (red_hits == 2 and blue_hits == 2):
            prize_level = 8  # 八等奖
        elif (red_hits == 3 and blue_hits == 0) or (red_hits == 1 and blue_hits == 2) or (red_hits == 2 and blue_hits == 1) or (red_hits == 0 and blue_hits == 2):
            prize_level = 9  # 九等奖
        
        if prize_level:
            prize_name = f"{['', '一', '二', '三', '四', '五', '六', '七', '八', '九'][prize_level]}等奖"
            prize_amount = PRIZE_TABLE.get(prize_level, 0)
            
            total_prize += prize_amount
            prize_counts[prize_name] = prize_counts.get(prize_name, 0) + 1
            
            winning_details.append({
                'ticket': ticket,
                'red_hits': red_hits,
                'blue_hits': blue_hits,
                'prize_level': prize_name,
                'prize_amount': prize_amount
            })
        
    return total_prize, prize_counts, winning_details

def format_detailed_winning_report(detailed_stats: Dict, prize_red: List, prize_blue: List) -> List[str]:
    """格式化详细中奖信息为报告字符串"""
    lines = [f"=== 详细中奖统计 (开奖号码: 红球{prize_red} 蓝球{prize_blue}) ==="]
    
    # 总体统计
    total_stats = detailed_stats['total']
    lines.append(f"总投注: {total_stats['tickets_count']}注, 总成本: {total_stats['cost_amount']:,}元")
    lines.append(f"总奖金: {total_stats['prize_amount']:,}元, 净利润: {total_stats['net_profit']:+,}元")
    lines.append(f"总回报率: {total_stats['roi_percent']:+.2f}%")
    
    if total_stats['prize_amount'] == 0:
        lines.append("本期推荐未中奖。")
        return lines
    
    # 单式投注统计
    single_stats = detailed_stats['single']
    if single_stats['tickets_count'] > 0:
        lines.append("")
        lines.append("--- 单式投注统计 ---")
        lines.append(f"投注数量: {single_stats['tickets_count']}注")
        lines.append(f"投注成本: {single_stats['cost_amount']:,}元 ({single_stats['tickets_count']}注 × {TICKET_PRICE}元)")
        lines.append(f"中奖金额: {single_stats['prize_amount']:,}元")
        lines.append(f"净利润: {single_stats['net_profit']:+,}元")
        lines.append(f"投资回报率: {single_stats['roi_percent']:+.2f}%")
        
        if single_stats['prize_counts']:
            prize_summary = ", ".join([f"{k}:{v}次" for k, v in single_stats['prize_counts'].items()])
            lines.append(f"中奖分布: {prize_summary}")
            
            # 显示单式中奖详情（限制显示数量）
            if single_stats['winning_details']:
                lines.append("中奖详情:")
                for i, win in enumerate(single_stats['winning_details'][:10]):  # 最多显示10条
                    red_nums, blue_nums = win['ticket']
                    lines.append(f"  单式第{i+1}注: 红球{red_nums} 蓝球{blue_nums} | "
                                f"命中{win['red_hits']}+{win['blue_hits']} | "
                                f"{win['prize_level']} | {win['prize_amount']:,}元")
                if len(single_stats['winning_details']) > 10:
                    lines.append(f"  ... 还有{len(single_stats['winning_details'])-10}注中奖")
        else:
            lines.append("中奖分布: 未中奖")
    
    # 复式投注统计
    complex_stats = detailed_stats['complex']
    if complex_stats['tickets_count'] > 0:
        lines.append("")
        lines.append("--- 复式投注统计 ---")
        complex_info = complex_stats['info']
        if complex_info:
            lines.append(f"复式选号: 红球{len(complex_info['red_numbers'])}个{complex_info['red_numbers']}, "
                        f"蓝球{len(complex_info['blue_numbers'])}个{complex_info['blue_numbers']}")
            lines.append(f"展开注数: {complex_info['total_combinations']}注")
        else:
            lines.append(f"投注数量: {complex_stats['tickets_count']}注")
        
        lines.append(f"投注成本: {complex_stats['cost_amount']:,}元 ({complex_stats['tickets_count']}注 × {TICKET_PRICE}元)")
        lines.append(f"中奖金额: {complex_stats['prize_amount']:,}元")
        lines.append(f"净利润: {complex_stats['net_profit']:+,}元")
        lines.append(f"投资回报率: {complex_stats['roi_percent']:+.2f}%")
        
        if complex_stats['prize_counts']:
            prize_summary = ", ".join([f"{k}:{v}次" for k, v in complex_stats['prize_counts'].items()])
            lines.append(f"中奖分布: {prize_summary}")
            
            # 显示复式中奖详情（限制显示数量）
            if complex_stats['winning_details']:
                lines.append("中奖详情:")
                for i, win in enumerate(complex_stats['winning_details'][:10]):  # 最多显示10条
                    red_nums, blue_nums = win['ticket']
                    lines.append(f"  复式第{i+1}注: 红球{red_nums} 蓝球{blue_nums} | "
                                f"命中{win['red_hits']}+{win['blue_hits']} | "
                                f"{win['prize_level']} | {win['prize_amount']:,}元")
                if len(complex_stats['winning_details']) > 10:
                    lines.append(f"  ... 还有{len(complex_stats['winning_details'])-10}注中奖")
        else:
            lines.append("中奖分布: 未中奖")
    
    return lines

def format_winning_tickets_for_report(winning_list: List[Dict], prize_red: List, prize_blue: List) -> List[str]:
    """格式化中奖信息为报告字符串（保持向后兼容）"""
    if not winning_list:
        return ["本期推荐未中奖。"]
    
    lines = [f"=== 中奖详情 (开奖号码: 红球{prize_red} 蓝球{prize_blue}) ==="]
    for win in winning_list:
        red_nums, blue_nums = win['ticket']
        lines.append(f"中奖注: 红球{red_nums} 蓝球{blue_nums} | "
                    f"命中{win['red_hits']}+{win['blue_hits']} | "
                    f"{win['prize_level']} | {win['prize_amount']:,}元")
    return lines

def manage_report(new_entry: Optional[Dict] = None, new_error: Optional[str] = None):
    """管理主报告文件，添加新记录或错误，并维护记录数量限制"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, MAIN_REPORT_FILE)
    
    # 读取现有记录
    existing_entries = []
    error_logs = []
    
    if os.path.exists(report_path):
        content = robust_file_read(report_path)
        if content:
            # 简单的解析逻辑：按分隔符分割记录
            sections = content.split("=" * 50)
            for section in sections:
                if "ERROR:" in section:
                    error_logs.append(section.strip())
                elif section.strip():
                    existing_entries.append(section.strip())
    
    # 添加新记录
    if new_entry:
        entry_lines = [
            f"评估时间: {new_entry['timestamp']}",
            f"评估期号: {new_entry['period']}",
            f"开奖号码: 红球{new_entry['winning_red']} 蓝球{new_entry['winning_blue']}",
        ]
        
        # 显示投注详情
        single_bets = new_entry.get('single_bets', 0)
        complex_bets = new_entry.get('complex_bets', 0)
        total_bets = new_entry.get('total_bets', single_bets + complex_bets)
        
        if complex_bets > 0:
            entry_lines.append(f"投注统计: 单式{single_bets}注, 复式{complex_bets}注, 合计{total_bets}注")
            
            # 显示复式信息
            complex_info = new_entry.get('complex_info')
            if complex_info:
                entry_lines.append(f"复式详情: 红球{len(complex_info['red_numbers'])}个, 蓝球{len(complex_info['blue_numbers'])}个")
        else:
            entry_lines.append(f"总投注: {total_bets}注")
        
        # 显示投资回报率信息
        detailed_stats = new_entry.get('detailed_stats')
        if detailed_stats:
            total_stats = detailed_stats['total']
            entry_lines.extend([
                f"总成本: {total_stats['cost_amount']:,}元",
                f"总奖金: {new_entry['total_prize']:,}元",
                f"净利润: {total_stats['net_profit']:+,}元",
                f"总回报率: {total_stats['roi_percent']:+.2f}%",
                f"中奖统计: {new_entry['prize_summary']}",
                ""
            ])
            
            # 分别显示单式和复式的投资回报
            single_stats = detailed_stats['single']
            complex_stats = detailed_stats['complex']
            
            if single_stats['tickets_count'] > 0:
                entry_lines.extend([
                    "--- 单式投资回报 ---",
                    f"单式成本: {single_stats['cost_amount']:,}元, 奖金: {single_stats['prize_amount']:,}元",
                    f"单式回报率: {single_stats['roi_percent']:+.2f}%"
                ])
            
            if complex_stats['tickets_count'] > 0:
                entry_lines.extend([
                    "--- 复式投资回报 ---",
                    f"复式成本: {complex_stats['cost_amount']:,}元, 奖金: {complex_stats['prize_amount']:,}元", 
                    f"复式回报率: {complex_stats['roi_percent']:+.2f}%"
                ])
            
            entry_lines.append("")
        else:
            # 向后兼容，如果没有detailed_stats
            entry_lines.extend([
                f"总奖金: {new_entry['total_prize']:,}元",
                f"中奖统计: {new_entry['prize_summary']}",
                ""
            ])
        
        if new_entry.get('winning_details'):
            entry_lines.extend(new_entry['winning_details'])
        
        new_entry_str = "\n".join(entry_lines)
        existing_entries.insert(0, new_entry_str)  # 插入到最前面
    
    if new_error:
        error_entry = f"ERROR: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {new_error}"
        error_logs.insert(0, error_entry)
    
    # 限制记录数量
    existing_entries = existing_entries[:MAX_NORMAL_RECORDS]
    error_logs = error_logs[:MAX_ERROR_LOGS]
    
    # 写回文件
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("大乐透预测验证报告\n")
            f.write("=" * 50 + "\n\n")
            
            for i, entry in enumerate(existing_entries, 1):
                f.write(f"记录 {i}:\n")
                f.write(entry)
                f.write("\n" + "=" * 50 + "\n\n")
            
            if error_logs:
                f.write("错误日志:\n")
                f.write("-" * 30 + "\n")
                for error in error_logs:
                    f.write(error + "\n")
                f.write("-" * 30 + "\n")
        
        log_message(f"报告已更新: {report_path}")
    except Exception as e:
        log_message(f"写入报告文件失败: {e}", "ERROR")

def main_process():
    """主处理流程"""
    try:
        log_message("开始大乐透预测验证流程...")
        
        # 读取CSV数据
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, CSV_FILE)
        csv_content = robust_file_read(csv_path)
        if not csv_content:
            raise Exception(f"无法读取CSV文件: {csv_path}")
        
        # 解析期号数据
        period_map, periods_list = get_period_data_from_csv(csv_content)
        if not period_map or len(periods_list) < 2:
            raise Exception("CSV数据不足或解析失败")
        
        # 确定评估期（要验证的已开奖期号）
        latest_period = periods_list[-1]
        
        log_message(f"要验证的期号: {latest_period}")
        
        # 查找预测该期号的分析报告
        report_file = find_matching_report(latest_period)
        if not report_file:
            raise Exception(f"未找到预测期号 {latest_period} 的分析报告")
        
        # 解析推荐号码
        report_content = robust_file_read(report_file)
        if not report_content:
            raise Exception("无法读取分析报告内容")
        
        rec_tickets, complex_reds, complex_blues = parse_recommendations_from_report(report_content)
        
        # 生成复式投注
        complex_tickets = []
        complex_info = None
        if complex_reds and complex_blues:
            complex_tickets = generate_complex_tickets(complex_reds, complex_blues)
            if complex_tickets:
                complex_info = {
                    'red_numbers': complex_reds,
                    'blue_numbers': complex_blues,
                    'total_combinations': len(complex_tickets)
                }
                log_message(f"复式号码解析成功: 红球{len(complex_reds)}个{complex_reds}, 蓝球{len(complex_blues)}个{complex_blues}")
                log_message(f"复式展开为{len(complex_tickets)}注单式投注")
        
        # 合并所有投注
        all_tickets = rec_tickets + complex_tickets
        if not all_tickets:
            raise Exception("未能解析到任何推荐号码")
        
        log_message(f"总投注统计: 单式{len(rec_tickets)}注, 复式{len(complex_tickets)}注, 合计{len(all_tickets)}注")
        
        # 获取开奖结果
        prize_data = period_map[latest_period]
        prize_red = prize_data['red']
        prize_blue = prize_data['blue']
        
        log_message(f"开奖结果: 红球{prize_red} 蓝球{prize_blue}")
        
        # 使用详细计算函数分别计算单式和复式的中奖情况
        total_prize, prize_counts, winning_details, detailed_stats = calculate_prize_detailed(
            rec_tickets, complex_tickets, complex_info, prize_red, prize_blue)
        
        # 格式化结果
        prize_summary = ", ".join([f"{k}:{v}次" for k, v in prize_counts.items()]) if prize_counts else "未中奖"
        winning_lines = format_detailed_winning_report(detailed_stats, prize_red, prize_blue)
        
        # 生成报告记录
        report_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': latest_period,
            'winning_red': prize_red,
            'winning_blue': prize_blue,
            'total_bets': len(all_tickets),
            'single_bets': len(rec_tickets),
            'complex_bets': len(complex_tickets),
            'complex_info': complex_info,
            'total_prize': total_prize,
            'prize_summary': prize_summary,
            'winning_details': winning_lines,
            'detailed_stats': detailed_stats
        }
        
        # 更新报告
        manage_report(new_entry=report_entry)
        
        # 详细的验证完成日志
        total_stats = detailed_stats['total']
        single_stats = detailed_stats['single']
        complex_stats = detailed_stats['complex']
        
        if complex_tickets:
            log_message(f"验证完成: 单式{len(rec_tickets)}注, 复式{len(complex_tickets)}注, 合计{len(all_tickets)}注")
            if complex_info:
                log_message(f"复式统计: 红球{len(complex_info['red_numbers'])}个, 蓝球{len(complex_info['blue_numbers'])}个, 展开{complex_info['total_combinations']}注")
        else:
            log_message(f"验证完成: 总投注{len(all_tickets)}注")
        
        log_message(f"总成本: {total_stats['cost_amount']:,}元, 总奖金: {total_prize:,}元, 总回报率: {total_stats['roi_percent']:+.2f}%")
        log_message(f"中奖分布: {prize_summary}")
        
        # 分别显示单式和复式的投资回报情况
        if single_stats['tickets_count'] > 0:
            log_message(f"单式投资: 成本{single_stats['cost_amount']:,}元, 奖金{single_stats['prize_amount']:,}元, 回报率{single_stats['roi_percent']:+.2f}%")
            if single_stats['prize_amount'] > 0:
                single_summary = ", ".join([f"{k}:{v}次" for k, v in single_stats['prize_counts'].items()])
                log_message(f"单式中奖: {single_summary}")
        
        if complex_stats['tickets_count'] > 0:
            log_message(f"复式投资: 成本{complex_stats['cost_amount']:,}元, 奖金{complex_stats['prize_amount']:,}元, 回报率{complex_stats['roi_percent']:+.2f}%")
            if complex_stats['prize_amount'] > 0:
                complex_summary = ", ".join([f"{k}:{v}次" for k, v in complex_stats['prize_counts'].items()])
                log_message(f"复式中奖: {complex_summary}")
        
        # 发送微信推送
        try:
            from wxPusher import send_verification_report
            log_message("正在发送验证报告微信推送...")
            
            # 构建验证数据
            verification_data = {
                'period': latest_period,
                'winning_red': prize_red,
                'winning_blue': prize_blue,
                'total_bets': len(all_tickets),
                'single_bets': len(rec_tickets),
                'complex_bets': len(complex_tickets),
                'complex_info': complex_info,
                'total_prize': total_prize,
                'prize_summary': prize_summary,
                'detailed_stats': detailed_stats
            }
            
            # 发送验证报告推送
            push_result = send_verification_report(verification_data)
            
            if push_result.get("success", False):
                log_message("✅ 验证报告微信推送发送成功")
            else:
                log_message(f"⚠️ 验证报告微信推送发送失败: {push_result.get('error', '未知错误')}", "WARNING")
                
        except ImportError:
            log_message("微信推送模块未找到，跳过推送功能", "WARNING")
        except Exception as e:
            log_message(f"微信推送发送异常: {e}", "ERROR")
        
    except Exception as e:
        error_msg = f"验证流程异常: {str(e)}"
        log_message(error_msg, "ERROR")
        manage_report(new_error=error_msg)
        log_message(f"详细错误:\n{traceback.format_exc()}", "ERROR")

if __name__ == "__main__":
    main_process() 