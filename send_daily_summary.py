# -*- coding: utf-8 -*-
"""
大乐透每日摘要推送脚本
==================

用于在GitHub Actions工作流结束时发送运行状态摘要到微信
"""

import os
import sys
import glob
import re
from datetime import datetime
from wxPusher import send_daily_summary, send_error_notification, send_analysis_report, send_verification_report

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
    """从分析报告中解析预测结果"""
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析预测期号 - 修复正则表达式
        period_match = re.search(r'第\s*(\d+)\s*期\s*号\s*码\s*推\s*荐', content)
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
        
        # 解析复式参考 - 修复正则表达式
        complex_red = []
        complex_blue = []
        
        # 查找复式参考部分 - 新的格式
        red_match = re.search(r'红球\s*\(Top\s*\d+\):\s*([\d\s]+)', content)
        blue_match = re.search(r'蓝球\s*\(Top\s*\d+\):\s*([\d\s]+)', content)
        
        if red_match:
            complex_red = [f'{int(n):02d}' for n in red_match.group(1).split()]
        if blue_match:
            complex_blue = [f'{int(n):02d}' for n in blue_match.group(1).split()]
        
        return period, recommendations, complex_red, complex_blue, content
    except Exception as e:
        print(f"解析预测报告失败: {e}")
        return None, [], [], [], ""

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

def get_error_summary():
    """从日志文件中提取错误摘要"""
    # 检查验证报告中的错误
    if check_file_exists("latest_dlt_calculation.txt"):
        try:
            with open("latest_dlt_calculation.txt", 'r', encoding='utf-8') as f:
                content = f.read()
                if "ERROR:" in content:
                    # 提取最新的错误信息
                    error_lines = [line for line in content.split('\n') if line.startswith("ERROR:")]
                    if error_lines:
                        return error_lines[-1].replace("ERROR:", "").strip()
        except:
            pass
    
    return None

def main():
    """主函数"""
    print("正在检查运行状态并发送每日摘要...")
    
    # 检查各个关键文件的状态
    analysis_success = False
    verification_success = False
    analysis_file = None
    error_msg = None
    
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
        
        # 检查是否有错误
        error_msg = get_error_summary()
        if error_msg:
            print(f"⚠️ 发现错误: {error_msg}")
    else:
        print("❌ 验证任务失败或未完成")
    
    # 检查数据文件更新
    if check_file_exists("daletou.csv"):
        print("✅ 数据文件已更新")
    else:
        print("❌ 数据文件缺失")
        if not error_msg:
            error_msg = "数据文件daletou.csv缺失"
    
    # 发送具体内容推送
    try:
        success_count = 0
        total_pushes = 0
        
        # 1. 发送预测报告（如果分析成功）
        if analysis_success and analysis_file:
            period, recommendations, complex_red, complex_blue, content = parse_predictions_from_report(analysis_file)
            if period and recommendations:
                result = send_analysis_report(content, period, recommendations, complex_red, complex_blue)
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
        
        # 3. 发送状态摘要（作为补充）
        result = send_daily_summary(
            analysis_success=analysis_success,
            verification_success=verification_success,
            analysis_file=os.path.basename(analysis_file) if analysis_file else None,
            error_msg=error_msg
        )
        total_pushes += 1
        if result.get("success", False):
            success_count += 1
            print("✅ 状态摘要推送成功")
        else:
            print(f"❌ 状态摘要推送失败: {result.get('error', '未知错误')}")
        
        # 如果有错误，发送错误通知
        if error_msg:
            try:
                send_error_notification(error_msg, "大乐透AI预测系统")
                print("✅ 错误通知已发送")
            except Exception as e:
                print(f"❌ 错误通知发送失败: {e}")
        
        # 总结推送结果
        if success_count == total_pushes:
            print(f"✅ 所有推送任务完成 ({success_count}/{total_pushes})")
            return 0
        else:
            print(f"⚠️ 部分推送任务失败 ({success_count}/{total_pushes})")
            return 1
            
    except Exception as e:
        print(f"❌ 推送异常: {e}")
        try:
            # 尝试发送错误通知
            send_error_notification(f"推送系统异常: {str(e)}", "摘要推送脚本")
        except:
            pass
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 