# -*- coding: utf-8 -*-
"""
大乐透数据处理器
================

本脚本负责从网络上获取大乐透的历史开奖数据，并将其与本地的CSV文件合并，
最终生成一个全面、更新的数据文件。

主要功能:
1.  从文本文件 (dlt_asc.txt) 获取包含开奖日期的完整历史数据。
2.  从HTML网页抓取最新的开奖数据（可能不含日期），作为补充。
3.  将两种来源的数据智能合并到主CSV文件 ('daletou.csv') 中。
    - 优先使用文本文件中的数据（尤其是日期）。
    - 能够处理新旧数据，自动去重和更新。
4.  具备良好的错误处理和日志记录能力，能应对网络波动和数据格式问题。
"""

import pandas as pd
import sys
import os
import requests
from bs4 import BeautifulSoup
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timedelta

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 获取脚本所在的目录，确保路径的相对性
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 目标CSV文件的完整路径。此文件将存储所有处理和合并后的数据。
# 您可以根据需要修改文件名，例如改为 'dlt_results.csv'。
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'daletou.csv')

# 网络数据源URL
# TXT源：提供包括日期在内的全量历史数据
TXT_DATA_URL = 'https://data.17500.cn/dlt_asc.txt'
# HTML源：提供最新的开奖数据，通常用于快速更新（但不含日期）
HTML_DATA_URL = "https://www.17500.cn/chart/dlt-tjb.html"

# 配置日志系统，用于跟踪脚本运行状态和错误信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
    ]
)
logger = logging.getLogger('dlt_data_processor')


# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

class SuppressOutput:
    """
    一个上下文管理器，用于临时抑制标准输出和/或捕获标准错误。
    这在调用会产生大量无关输出的库函数时非常有用。
    捕获的错误信息会通过日志系统记录下来，避免信息丢失。
    """
    def __init__(self, suppress_stdout: bool = True, capture_stderr: bool = True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.old_stdout = None
        self.old_stderr = None
        self.stderr_io = io.StringIO()

    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')

        if self.capture_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.stderr_io
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复标准错误
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr = self.stderr_io.getvalue()
            if captured_stderr.strip():
                logger.warning(f"在一个被抑制的输出块中捕获到标准错误:\n{captured_stderr.strip()}")
            self.stderr_io.close()

        # 恢复标准输出
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed:
                sys.stdout.close()
            sys.stdout = self.old_stdout

        return False  # 不抑制任何发生的异常


# ==============================================================================
# --- 数据获取模块 ---
# ==============================================================================

def infer_date_from_period(period: str, existing_data: list = None) -> str:
    """
    根据期号推算日期。大乐透每周一、三、六开奖。
    
    Args:
        period (str): 期号，格式如 "25042" 
        existing_data (list): 现有数据，用于获取参考日期
        
    Returns:
        str: 推算的日期，格式为 YYYY-MM-DD
    """
    try:
        # 解析期号：前2位是年份（2025 -> 25），后3位是期号
        if len(period) == 5:
            year = 2000 + int(period[:2])
            period_num = int(period[2:])
        else:
            # 对于较早期号，可能是其他格式
            return ""
            
        # 每年大约有156期（一年52周 × 3期/周）
        # 每期间隔约2-3天，这里使用平均2.33天（一年365天/156期）
        
        # 2025年第一期的日期是2025-01-01
        if year == 2025:
            base_date = datetime(2025, 1, 1)
            # 计算从第一期到当前期的天数
            days_offset = (period_num - 1) * 2.33  # 平均每期间隔2.33天
            
            # 调整到最近的开奖日（周一、三、六）
            target_date = base_date + timedelta(days=int(days_offset))
            
            # 调整到最近的开奖日
            weekday = target_date.weekday()  # 0=Monday, 2=Wednesday, 5=Saturday
            if weekday not in [0, 2, 5]:
                # 找到最近的开奖日
                days_to_next = min(
                    (0 - weekday) % 7,  # 到下周一
                    (2 - weekday) % 7,  # 到本周三或下周三
                    (5 - weekday) % 7   # 到本周六或下周六
                )
                if days_to_next > 3:  # 如果距离太远，找前一个开奖日
                    days_to_prev = min(
                        (weekday - 0) if weekday > 0 else 7,  # 到上周一
                        (weekday - 2) if weekday > 2 else 7,  # 到上周三  
                        (weekday - 5) if weekday > 5 else 7   # 到上周六
                    )
                    target_date = target_date - timedelta(days=days_to_prev)
                else:
                    target_date = target_date + timedelta(days=days_to_next)
            
            return target_date.strftime('%Y-%m-%d')
            
        # 对于其他年份，可以根据需要扩展
        return ""
        
    except (ValueError, IndexError):
        return ""


def fetch_latest_data_from_html(url: str = HTML_DATA_URL) -> list:
    """
    从指定的HTML网页抓取最新的大乐透数据。
    注意：此数据源通常不包含开奖日期，但我们会尝试推算日期。

    Args:
        url (str): 目标网页的URL。

    Returns:
        list: 一个包含字典的列表，每个字典代表一期数据，格式为
              {'期号': '...', '红球': '...', '蓝球': '...', '日期': '...'}。
              如果失败则返回空列表。
    """
    logger.info("正在从HTML网页抓取最新大乐透数据...")
    data = []
    try:
        session = requests.Session()
        session.trust_env = False  # 禁用环境变量中的代理，提高连接成功率

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # 如果请求失败 (如 404, 500)，则抛出异常

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        if not table:
            logger.warning("在网页中未能找到数据表格。")
            return []

        # 遍历表格的每一行
        for row in table.find_all('tr')[1:]:  # 跳过表头
            cells = row.find_all('td')
            # 数据行有效性校验
            if len(cells) < 3:
                continue

            try:
                # 提取期号
                period_text = cells[0].text.strip().replace("期", "")
                if not period_text.isdigit():
                    continue

                # 提取红球和蓝球（大乐透是5个红球+2个蓝球）
                red_balls_str = cells[1].text.strip().replace(" ", ",")
                blue_balls_str = cells[2].text.strip().replace(" ", ",")

                # 验证号码格式
                red_numbers = [int(x) for x in red_balls_str.split(',')]
                blue_numbers = [int(x) for x in blue_balls_str.split(',')]
                
                if len(red_numbers) != 5 or len(blue_numbers) != 2:
                    continue
                if not all(1 <= r <= 35 for r in red_numbers):
                    continue
                if not all(1 <= b <= 12 for b in blue_numbers):
                    continue

                # 推算日期
                inferred_date = infer_date_from_period(period_text)

                data.append({
                    '期号': period_text,
                    '红球': red_balls_str,
                    '蓝球': blue_balls_str,
                    '日期': inferred_date  # 添加推算的日期
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"解析表格行时出错: {row.text.strip()}. 错误: {e}. 跳过此行。")
                continue

        logger.info(f"从HTML网页成功获取 {len(data)} 期数据。")
        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"从HTML网页获取数据失败: {e}")
        return []


def fetch_full_data_from_txt(url: str = TXT_DATA_URL) -> list:
    """
    从指定的文本文件URL下载完整的历史数据。
    此数据源包含开奖日期，是数据的主要来源。

    Args:
        url (str): 目标 .txt 文件的URL。

    Returns:
        list: 包含文件中每一行字符串的列表。如果失败则返回空列表。
    """
    logger.info(f"正在从TXT文件源 ({url}) 下载全量数据...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'  # 显式设置编码
        data_lines = response.text.strip().split('\n')
        logger.info(f"成功下载 {len(data_lines)} 行数据。")
        return data_lines
    except requests.exceptions.RequestException as e:
        logger.error(f"从TXT文件源下载数据失败: {e}")
        return []


def parse_txt_data(data_lines: list) -> list:
    """
    解析从TXT文件获取的原始数据行，提取结构化的开奖信息。

    Args:
        data_lines (list): 包含原始数据行的列表。

    Returns:
        list: 包含解析后数据字典的列表，每个字典包含期号、日期、红球和蓝球信息。
    """
    logger.info("正在解析TXT文件数据...")
    structured_data = []
    
    for line_num, line in enumerate(data_lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            # 根据用户提供的大乐透数据格式解析
            # 数据格式：Seq 日期 红球1 红球2 红球3 红球4 红球5 蓝球1 蓝球2
            parts = line.split(',', 1)
            first_part = parts[0].strip()
            fields = first_part.split()
            
            if len(fields) < 9:
                logger.warning(f"跳过行 {line_num}（字段不足9个）：{line}")
                continue
            
            seq = fields[0]
            date = fields[1]  # 日期字段
            red_balls = fields[2:7]  # 提取5个红球
            blue_balls = fields[7:9]  # 提取2个蓝球
            
            # 验证数据有效性
            if not seq.isdigit():
                logger.warning(f"跳过行 {line_num}（期号无效）：{line}")
                continue
                
            # 验证红球
            try:
                red_nums = [int(r) for r in red_balls]
                if len(red_nums) != 5 or not all(1 <= r <= 35 for r in red_nums):
                    logger.warning(f"跳过行 {line_num}（红球无效）：{line}")
                    continue
            except ValueError:
                logger.warning(f"跳过行 {line_num}（红球格式错误）：{line}")
                continue
                
            # 验证蓝球
            try:
                blue_nums = [int(b) for b in blue_balls]
                if len(blue_nums) != 2 or not all(1 <= b <= 12 for b in blue_nums):
                    logger.warning(f"跳过行 {line_num}（蓝球无效）：{line}")
                    continue
            except ValueError:
                logger.warning(f"跳过行 {line_num}（蓝球格式错误）：{line}")
                continue
            
            structured_data.append({
                '期号': seq,
                '日期': date,
                '红球': ','.join(red_balls),
                '蓝球': ','.join(blue_balls)
            })
            
        except Exception as e:
            logger.warning(f"解析行 {line_num} 时出错：{e}。跳过此行：{line}")
            continue
    
    logger.info(f"成功解析 {len(structured_data)} 条有效记录。")
    return structured_data


def update_csv_file(csv_path: str, all_new_data: list):
    """
    将新数据与现有的CSV文件合并并更新。
    
    Args:
        csv_path (str): CSV文件的路径。
        all_new_data (list): 包含新数据字典的列表。
    """
    logger.info(f"正在更新CSV文件: {csv_path}")
    
    # 期号映射，用于快速查找和去重
    existing_periods = set()
    all_data = []
    
    # 如果CSV文件已存在，先读取现有数据
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if '期号' in row and row['期号'].strip():
                        existing_periods.add(row['期号'].strip())
                        all_data.append(row)
            logger.info(f"已读取现有数据 {len(all_data)} 条记录。")
        except Exception as e:
            logger.error(f"读取现有CSV文件失败: {e}")
    
    # 添加新数据（去重）
    new_count = 0
    updated_count = 0
    
    for new_record in all_new_data:
        period = new_record['期号'].strip()
        
        if period in existing_periods:
            # 更新现有记录（如果新记录有日期而旧记录没有）
            for i, existing_record in enumerate(all_data):
                if existing_record['期号'].strip() == period:
                    # 如果新记录有日期且旧记录没有日期，则更新
                    if new_record.get('日期') and not existing_record.get('日期'):
                        all_data[i].update(new_record)
                        updated_count += 1
                    break
        else:
            # 添加新记录
            all_data.append(new_record)
            existing_periods.add(period)
            new_count += 1
    
    # 按期号排序
    try:
        all_data.sort(key=lambda x: int(x['期号']))
    except ValueError:
        logger.warning("某些期号无法转换为整数，使用字符串排序。")
        all_data.sort(key=lambda x: x['期号'])
    
    # 写回CSV文件
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            if all_data:
                fieldnames = ['期号', '日期', '红球', '蓝球']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)
        
        logger.info(f"CSV文件更新成功：新增 {new_count} 条，更新 {updated_count} 条，总计 {len(all_data)} 条记录。")
    except Exception as e:
        logger.error(f"写入CSV文件失败: {e}")


# ==============================================================================
# --- 主程序执行函数 ---
# ==============================================================================

def fetch_dlt_data():
    """
    主函数：获取大乐透数据并更新本地CSV文件。
    参考用户提供的大乐透爬虫代码逻辑。
    """
    logger.info("开始大乐透数据获取和处理流程...")
    
    # 获取TXT格式的全量历史数据
    txt_data_lines = fetch_full_data_from_txt()
    txt_structured_data = []
    if txt_data_lines:
        txt_structured_data = parse_txt_data(txt_data_lines)
    
    # 获取HTML格式的最新数据作为补充
    html_data = fetch_latest_data_from_html()
    
    # 合并所有数据
    all_new_data = txt_structured_data + html_data
    
    if not all_new_data:
        logger.warning("未能获取到任何有效数据。请检查网络连接和数据源。")
        return
    
    # 更新CSV文件
    update_csv_file(CSV_FILE_PATH, all_new_data)
    logger.info("大乐透数据处理完成。")


if __name__ == "__main__":
    fetch_dlt_data() 