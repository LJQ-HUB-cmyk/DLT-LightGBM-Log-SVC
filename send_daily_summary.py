#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大乐透每日摘要推送脚本
==================

用于在GitHub Actions工作流结束时发送运行状态摘要到微信
"""

import os
import re
import glob
import sys
from wxPusher import send_analysis_report, send_verification_report

def check_file_exists(file_path):
    """检查文件是否存在"""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def get_latest_analysis_file():
    """获取最新的分析报告文件名"""
    pattern = "dlt_analysis_output_*.txt"
    files = glob.glob(pattern)
    if files:
        # 按文件名排序，取最新的
        latest_file = sorted(files)[-1]
        return latest_file
    return None

def parse_predictions_from_report(report_file):
    """从分析报告中解析预测结果和分析信息"""
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析预测期号 - 修复正则表达式，匹配实际格式
        period_match = re.search(r'第\s*(\d+)\s*期\s*号\s*码\s*推\s*荐', content)
        if not period_match:
            # 备用匹配：本次预测目标
            period_match = re.search(r'本次预测目标:\s*第\s*(\d+)\s*期', content)
        period = period_match.group(1) if period_match else "未知"
        
        # 解析所有推荐号码（不限制数量）
        recommendations = []
        # 匹配格式：注 1: 红球 [29 30 32 33 34] 蓝球 [02 07] (综合分: 535.85)
        rec_pattern = re.compile(r'注\s*(\d+):\s*红球\s*\[([\d\s]+)\]\s*蓝球\s*\[([\d\s]+)\]\s*\(综合分:\s*([\d.]+)\)')
        for match in rec_pattern.finditer(content):
            note_num = match.group(1)
            red_balls = ' '.join(f'{int(n):02d}' for n in match.group(2).split())
            blue_balls = ' '.join(f'{int(n):02d}' for n in match.group(3).split())
            score = match.group(4)
            recommendations.append(f"注{note_num}: 红球 {red_balls} 蓝球 {blue_balls} (分值:{score})")
        
        # 解析复式参考 - 修复正则表达式匹配实际格式
        complex_red = []
        complex_blue = []
        
        # 查找复式参考部分 - 匹配"红球 (平衡选择): xx xx xx"格式
        red_match = re.search(r'红球\s*\([^)]+\):\s*([\d\s]+)', content)
        blue_match = re.search(r'蓝球\s*\([^)]+\):\s*([\d\s]+)', content)
        
        if red_match:
            complex_red = [f'{int(n):02d}' for n in red_match.group(1).split()]
        if blue_match:
            complex_blue = [f'{int(n):02d}' for n in blue_match.group(1).split()]
        
        # 解析热冷号码分析
        hot_cold_info = {}
        
        # 解析红球热号
        red_hot_match = re.search(r'红球热号\(最近5期\):\s*([\d\s]+)\s*\(共(\d+)个\)', content)
        if red_hot_match:
            hot_cold_info['red_hot'] = red_hot_match.group(1).strip()
            hot_cold_info['red_hot_count'] = red_hot_match.group(2)
        
        # 解析红球冷号
        red_cold_match = re.search(r'红球冷号\(6期未出\):\s*([\d\s]+)\s*\(共(\d+)个\)', content)
        if red_cold_match:
            hot_cold_info['red_cold'] = red_cold_match.group(1).strip()
            hot_cold_info['red_cold_count'] = red_cold_match.group(2)
        
        # 解析蓝球热号
        blue_hot_match = re.search(r'蓝球热号\(最近5期\):\s*([\d\s]+)\s*\(共(\d+)个\)', content)
        if blue_hot_match:
            hot_cold_info['blue_hot'] = blue_hot_match.group(1).strip()
            hot_cold_info['blue_hot_count'] = blue_hot_match.group(2)
        
        # 解析蓝球冷号
        blue_cold_match = re.search(r'蓝球冷号\(6期未出\):\s*([\d\s]+)\s*\(共(\d+)个\)', content)
        if blue_cold_match:
            hot_cold_info['blue_cold'] = blue_cold_match.group(1).strip()
            hot_cold_info['blue_cold_count'] = blue_cold_match.group(2)
        
        # 解析分布信息
        distribution_info = {}
        red_dist_match = re.search(r'红球分布 - 小区\(1-12\):(\d+)个, 中区\(13-24\):(\d+)个, 大区\(25-35\):(\d+)个', content)
        if red_dist_match:
            distribution_info['red_small'] = red_dist_match.group(1)
            distribution_info['red_medium'] = red_dist_match.group(2)
            distribution_info['red_large'] = red_dist_match.group(3)
        
        blue_dist_match = re.search(r'蓝球分布 - 小号\(1-6\):(\d+)个, 大号\(7-12\):(\d+)个', content)
        if blue_dist_match:
            distribution_info['blue_small'] = blue_dist_match.group(1)
            distribution_info['blue_large'] = blue_dist_match.group(2)
        
        # 解析回测信息
        backtest_info = {}
        roi_match = re.search(r'投资回报率 \(ROI\):\s*([-\d.]+)%', content)
        if roi_match:
            backtest_info['roi'] = roi_match.group(1)
        
        period_match_bt = re.search(r'回测周期:\s*最近\s*(\d+)\s*期', content)
        if period_match_bt:
            backtest_info['test_periods'] = period_match_bt.group(1)
        
        return period, recommendations, complex_red, complex_blue, hot_cold_info, distribution_info, backtest_info, content
    except Exception as e:
        print(f"解析预测报告失败: {e}")
        return None, [], [], [], {}, {}, {}, ""

def parse_verification_from_calculation():
    """从验证报告中解析验证结果"""
    try:
        if not check_file_exists("latest_dlt_calculation.txt"):
            return None
            
        with open("latest_dlt_calculation.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到最新的验证记录（最后一条）
        verification_blocks = re.findall(
            r'期号:\s*(\d+).*?开奖号码.*?红球:\s*([\d\s]+).*?蓝球:\s*([\d\s]+).*?总奖金:\s*([,\d]+).*?中奖统计:\s*([^\n]+)',
            content, re.DOTALL
        )
        
        if verification_blocks:
            latest = verification_blocks[-1]  # 最后一条记录
            period = latest[0]
            winning_red = [int(n) for n in latest[1].split()]
            winning_blue = [int(n) for n in latest[2].split()]
            total_prize = int(latest[3].replace(',', ''))
            prize_summary = latest[4].strip()
            
            return {
                'period': period,
                'winning_red': winning_red,
                'winning_blue': winning_blue,
                'total_prize': total_prize,
                'prize_summary': prize_summary,
                'total_bets': 10  # 假设10注，实际应该从报告中解析
            }
    except Exception as e:
        print(f"解析验证报告失败: {e}")
    
    return None

def main():
    """主函数"""
    print("正在检查运行状态并发送核心推送...")
    
    # 检查各个关键文件的状态
    analysis_success = False
    verification_success = False
    analysis_file = None
    
    # 检查分析是否成功
    analysis_file = get_latest_analysis_file()
    if analysis_file and check_file_exists("latest_dlt_analysis.txt"):
        analysis_success = True
        print(f"✅ 分析任务完成: {analysis_file}")
    else:
        print("❌ 分析任务失败或未完成")
    
    # 检查验证是否成功
    verification_data = None
    if check_file_exists("latest_dlt_calculation.txt"):
        verification_success = True
        verification_data = parse_verification_from_calculation()
        print("✅ 验证任务完成")
    else:
        print("❌ 验证任务失败或未完成")
    
    # 检查数据文件更新
    if check_file_exists("daletou.csv"):
        print("✅ 数据文件已更新")
    else:
        print("❌ 数据文件缺失")
    
    # 只发送核心内容推送：预测报告和验证报告
    try:
        success_count = 0
        total_pushes = 0
        
        # 1. 发送预测报告（如果分析成功）
        if analysis_success and analysis_file:
            period, recommendations, complex_red, complex_blue, hot_cold_info, distribution_info, backtest_info, content = parse_predictions_from_report(analysis_file)
            if period and recommendations:
                result = send_analysis_report(
                    content, period, recommendations, complex_red, complex_blue, 
                    hot_cold_info, distribution_info, backtest_info
                )
                total_pushes += 1
                if result.get("success", False):
                    success_count += 1
                    print("✅ 预测报告推送成功")
                else:
                    print(f"❌ 预测报告推送失败: {result.get('error', '未知错误')}")
        
        # 2. 发送验证报告（如果验证成功）
        if verification_success and verification_data:
            result = send_verification_report(verification_data)
            total_pushes += 1
            if result.get("success", False):
                success_count += 1
                print("✅ 验证报告推送成功")
            else:
                print(f"❌ 验证报告推送失败: {result.get('error', '未知错误')}")
        
        # 总结推送结果
        if success_count == total_pushes and total_pushes > 0:
            print(f"✅ 所有核心推送任务完成 ({success_count}/{total_pushes})")
            return 0
        elif total_pushes == 0:
            print("ℹ️ 没有可推送的内容")
            return 0
        else:
            print(f"⚠️ 部分推送任务失败 ({success_count}/{total_pushes})")
            return 1
            
    except Exception as e:
        print(f"❌ 推送异常: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 