# -*- coding: utf-8 -*-
"""
大乐透微信推送模块
================

提供微信推送功能，用于推送大乐透分析报告和验证报告
"""

import requests
import logging
import json
import os
from datetime import datetime
from typing import Optional, List, Dict
from math import comb

# 微信推送配置
APP_TOKEN = "AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw"
USER_UIDS = ["UID_yYObqdMVScIa66DGR2n2PCRFL10w"]
TOPIC_IDS = [39909]  # 大乐透专用主题ID

def send_wxpusher_message(content: str, title: str = None, topicIds: List[int] = None, uids: List[str] = None) -> Dict:
    """发送微信推送消息
    
    Args:
        content: 消息内容
        title: 消息标题
        topicIds: 主题ID列表，默认使用全局配置
        uids: 用户ID列表，默认使用全局配置
    
    Returns:
        API响应结果字典
    """
    url = "https://wxpusher.zjiecode.com/api/send/message"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "appToken": APP_TOKEN,
        "content": content,
        "uids": uids or USER_UIDS,
        "topicIds": topicIds or TOPIC_IDS,
        "summary": title or "大乐透推荐更新",
        "contentType": 1,  # 1=文本，2=HTML
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result.get("success", False):
            logging.info(f"微信推送成功: {title}")
            return {"success": True, "data": result}
        else:
            logging.error(f"微信推送失败: {result.get('msg', '未知错误')}")
            return {"success": False, "error": result.get('msg', '推送失败')}
            
    except requests.exceptions.RequestException as e:
        logging.error(f"微信推送网络错误: {e}")
        return {"success": False, "error": f"网络错误: {str(e)}"}
    except Exception as e:
        logging.error(f"微信推送异常: {e}")
        return {"success": False, "error": f"未知异常: {str(e)}"}

def send_analysis_report(report_content: str, period: int, recommendations: List[str], 
                         complex_red: List[str] = None, complex_blue: List[str] = None,
                         hot_cold_info: Dict = None, distribution_info: Dict = None, 
                         backtest_info: Dict = None) -> Dict:
    """发送大乐透分析报告
    
    Args:
        report_content: 报告内容（文件名或内容）
        period: 预测期号
        recommendations: 推荐号码列表
        complex_red: 复式红球列表（可选）
        complex_blue: 复式蓝球列表（可选）
        hot_cold_info: 热冷号码分析信息（可选）
        distribution_info: 号码分布分析信息（可选）
        backtest_info: 回测性能信息（可选）
    
    Returns:
        推送结果字典
    """
    title = f"🎯 大乐透第{period}期预测报告"
    
    # 提取关键信息制作完整版推送
    try:
        # 构建单式推荐内容 - 显示所有推荐号码
        rec_summary = ""
        if recommendations:
            for i, rec in enumerate(recommendations):
                rec_summary += f"{rec}\n"
                # 每5注换行一次，便于阅读
                if (i + 1) % 5 == 0 and i < len(recommendations) - 1:
                    rec_summary += "\n"
        
        # 构建复式参考内容
        complex_summary = ""
        if complex_red and complex_blue:
            # 计算复式组合数：C(红球数,5) * C(蓝球数,2)
            red_combinations = comb(len(complex_red), 5) if len(complex_red) >= 5 else 0
            blue_combinations = comb(len(complex_blue), 2) if len(complex_blue) >= 2 else 0
            total_combinations = red_combinations * blue_combinations
            
            complex_summary = f"""
📦 复式参考：
红球({len(complex_red)}个): {' '.join(complex_red)}
蓝球({len(complex_blue)}个): {' '.join(complex_blue)}

💡 复式共可组成 {total_combinations:,} 注
💰 投注成本: {total_combinations * 3:,} 元(单注3元)"""
        
        # 构建热冷号码分析内容
        hot_cold_summary = ""
        if hot_cold_info:
            hot_cold_summary = f"""
📈 热冷号码分析：
🔥 红球热号({hot_cold_info.get('red_hot_count', 0)}个): {hot_cold_info.get('red_hot', '')}
❄️ 红球冷号({hot_cold_info.get('red_cold_count', 0)}个): {hot_cold_info.get('red_cold', '')}
🔥 蓝球热号({hot_cold_info.get('blue_hot_count', 0)}个): {hot_cold_info.get('blue_hot', '')}
❄️ 蓝球冷号({hot_cold_info.get('blue_cold_count', 0)}个): {hot_cold_info.get('blue_cold', '')}"""

        # 构建分布分析内容
        distribution_summary = ""
        if distribution_info:
            distribution_summary = f"""
📊 号码分布分析：
红球分布: 小区{distribution_info.get('red_small', 0)}个 | 中区{distribution_info.get('red_medium', 0)}个 | 大区{distribution_info.get('red_large', 0)}个
蓝球分布: 小号{distribution_info.get('blue_small', 0)}个 | 大号{distribution_info.get('blue_large', 0)}个"""

        # 构建回测信息内容
        backtest_summary = ""
        if backtest_info:
            backtest_summary = f"""
📉 策略回测表现：
回测周期: 最近{backtest_info.get('test_periods', 'N/A')}期
投资回报率: {backtest_info.get('roi', 'N/A')}%"""

        # 构建完整推送内容
        content = f"""🎯 大乐透第{period}期AI智能预测

📊 单式推荐 (共{len(recommendations)}注)：
{rec_summary.strip()}
{complex_summary}
{hot_cold_summary}
{distribution_summary}
{backtest_summary}

⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

💡 仅供参考，理性投注！祝您好运！"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建分析报告推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def send_verification_report(verification_data: Dict) -> Dict:
    """发送大乐透验证报告
    
    Args:
        verification_data: 验证报告数据字典，包含中奖信息
    
    Returns:
        推送结果字典
    """
    try:
        period = verification_data.get('period', '未知')
        title = f"✅ 大乐透第{period}期验证报告"
        
        winning_red = verification_data.get('winning_red', [])
        winning_blue = verification_data.get('winning_blue', [])
        total_bets = verification_data.get('total_bets', 0)
        total_prize = verification_data.get('total_prize', 0)
        prize_summary = verification_data.get('prize_summary', '未中奖')
        
        # 构建完整验证报告内容
        content = f"""✅ 大乐透第{period}期开奖验证

🎱 开奖号码：
红球：{' '.join(f'{n:02d}' for n in winning_red)}
蓝球：{' '.join(f'{n:02d}' for n in winning_blue)}

📊 验证结果：
投注总数：{total_bets}注
中奖统计：{prize_summary}
总奖金：{total_prize:,}元

💰 投资回报：
成本：{total_bets * 3:,}元（单注3元）
收益：{total_prize - total_bets * 3:,}元
回报率：{((total_prize - total_bets * 3) / (total_bets * 3) * 100):.2f}%

⏰ 验证时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建验证报告推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def send_error_notification(error_msg: str, script_name: str = "大乐透系统") -> Dict:
    """发送错误通知
    
    Args:
        error_msg: 错误信息
        script_name: 脚本名称
    
    Returns:
        推送结果字典
    """
    title = f"⚠️ {script_name}运行异常"
    
    content = f"""⚠️ 系统运行异常通知

📍 异常位置：{script_name}
🕒 发生时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
❌ 错误信息：
{error_msg}

请及时检查系统状态！"""
    
    return send_wxpusher_message(content, title)

def send_daily_summary(analysis_success: bool, verification_success: bool, 
                      analysis_file: str = None, error_msg: str = None) -> Dict:
    """发送每日运行摘要
    
    Args:
        analysis_success: 分析是否成功
        verification_success: 验证是否成功
        analysis_file: 分析报告文件名
        error_msg: 错误信息（如有）
    
    Returns:
        推送结果字典
    """
    title = "📊 大乐透系统日报"
    
    # 状态图标
    analysis_status = "✅" if analysis_success else "❌"
    verification_status = "✅" if verification_success else "❌"
    
    content = f"""📊 大乐透AI预测系统日报

🕒 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

📈 任务执行状态：
{analysis_status} 数据分析与预测
{verification_status} 历史验证计算

📁 生成文件："""
    
    if analysis_file:
        content += f"\n• {analysis_file}"
    
    if error_msg:
        content += f"\n\n⚠️ 异常信息：\n{error_msg}"
    
    content += "\n\n🔔 系统已自动完成定时任务"
    
    return send_wxpusher_message(content, title)

def test_wxpusher_connection() -> bool:
    """测试微信推送连接
    
    Returns:
        连接是否成功
    """
    test_content = f"🔧 大乐透推送系统测试\n\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n如收到此消息，说明推送功能正常！"
    result = send_wxpusher_message(test_content, "🔧 推送测试")
    return result.get("success", False)

if __name__ == "__main__":
    # 测试推送功能
    print("正在测试大乐透微信推送功能...")
    
    # 测试基本推送
    if test_wxpusher_connection():
        print("✅ 微信推送测试成功！")
        
        # 测试分析报告推送
        test_recommendations = [
            "注 1: 红球 [01 05 12 25 35] 蓝球 [03 08]",
            "注 2: 红球 [02 08 15 28 33] 蓝球 [05 11]",
            "注 3: 红球 [03 10 18 30 34] 蓝球 [02 09]"
        ]
        
        print("测试分析报告推送...")
        send_analysis_report("测试报告内容", 2025069, test_recommendations)
        
        print("测试验证报告推送...")
        test_verification = {
            'period': 2025068,
            'winning_red': [1, 4, 17, 20, 22],
            'winning_blue': [4, 10],
            'total_bets': 15,
            'total_prize': 45,
            'prize_summary': '九等奖:3次'
        }
        send_verification_report(test_verification)
        
    else:
        print("❌ 微信推送测试失败！请检查配置。")