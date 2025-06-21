#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的大乐透组合生成策略
实现多种先进的号码组合生成方法
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
import itertools

class ImprovedCombinationGenerator:
    """改进的组合生成器"""
    
    def __init__(self, red_scores: Dict[int, float], blue_scores: Dict[int, float]):
        self.red_scores = red_scores
        self.blue_scores = blue_scores
        self.red_range = list(range(1, 36))  # 1-35
        self.blue_range = list(range(1, 13))  # 1-12
        
    def strategy_1_balanced_zone_selection(self, num_combinations: int = 12) -> List[Dict]:
        """
        策略1: 平衡分区选择法
        确保每个区间都有合理的号码分布
        """
        combinations = []
        
        # 定义红球三分区
        zone1 = list(range(1, 13))   # 1-12
        zone2 = list(range(13, 25))  # 13-24  
        zone3 = list(range(25, 36))  # 25-35
        
        # 定义蓝球大小分区
        blue_small = list(range(1, 7))   # 1-6
        blue_large = list(range(7, 13))  # 7-12
        
        # 获取各区间的Top候选
        zone1_candidates = sorted([(n, self.red_scores.get(n, 0)) for n in zone1], 
                                key=lambda x: x[1], reverse=True)[:8]
        zone2_candidates = sorted([(n, self.red_scores.get(n, 0)) for n in zone2], 
                                key=lambda x: x[1], reverse=True)[:8]
        zone3_candidates = sorted([(n, self.red_scores.get(n, 0)) for n in zone3], 
                                key=lambda x: x[1], reverse=True)[:8]
        
        blue_small_candidates = sorted([(n, self.blue_scores.get(n, 0)) for n in blue_small], 
                                     key=lambda x: x[1], reverse=True)[:4]
        blue_large_candidates = sorted([(n, self.blue_scores.get(n, 0)) for n in blue_large], 
                                     key=lambda x: x[1], reverse=True)[:4]
        
        # 定义多种分区组合模式
        zone_patterns = [
            (2, 2, 1),  # 小区2个，中区2个，大区1个
            (2, 1, 2),  # 小区2个，中区1个，大区2个
            (1, 2, 2),  # 小区1个，中区2个，大区2个
            (1, 3, 1),  # 小区1个，中区3个，大区1个
            (3, 1, 1),  # 小区3个，中区1个，大区1个
            (1, 1, 3),  # 小区1个，中区1个，大区3个
        ]
        
        blue_patterns = [
            (1, 1),  # 小号1个，大号1个
            (2, 0),  # 小号2个，大号0个
            (0, 2),  # 小号0个，大号2个
        ]
        
        used_combinations = set()
        
        for _ in range(num_combinations * 10):  # 多生成一些候选
            if len(combinations) >= num_combinations:
                break
                
            # 随机选择分区模式
            z1_count, z2_count, z3_count = random.choice(zone_patterns)
            blue_small_count, blue_large_count = random.choice(blue_patterns)
            
            try:
                # 从各区间随机选择
                selected_reds = []
                if z1_count > 0:
                    selected_reds.extend(random.sample([n for n, _ in zone1_candidates], 
                                                     min(z1_count, len(zone1_candidates))))
                if z2_count > 0:
                    selected_reds.extend(random.sample([n for n, _ in zone2_candidates], 
                                                     min(z2_count, len(zone2_candidates))))
                if z3_count > 0:
                    selected_reds.extend(random.sample([n for n, _ in zone3_candidates], 
                                                     min(z3_count, len(zone3_candidates))))
                
                # 蓝球选择
                selected_blues = []
                if blue_small_count > 0:
                    selected_blues.extend(random.sample([n for n, _ in blue_small_candidates], 
                                                      min(blue_small_count, len(blue_small_candidates))))
                if blue_large_count > 0:
                    selected_blues.extend(random.sample([n for n, _ in blue_large_candidates], 
                                                      min(blue_large_count, len(blue_large_candidates))))
                
                # 确保数量正确
                if len(selected_reds) == 5 and len(selected_blues) == 2:
                    combo_key = (tuple(sorted(selected_reds)), tuple(sorted(selected_blues)))
                    if combo_key not in used_combinations:
                        combinations.append({
                            'red': sorted(selected_reds),
                            'blue': sorted(selected_blues),
                            'strategy': '平衡分区'
                        })
                        used_combinations.add(combo_key)
                        
            except (ValueError, IndexError):
                continue
                
        return combinations
    
    def strategy_2_score_clustering(self, num_combinations: int = 12) -> List[Dict]:
        """
        策略2: 分数聚类法
        将相似分数的号码分组，避免过度集中
        """
        combinations = []
        
        # 对红球分数进行聚类
        red_scores_list = [(num, score) for num, score in self.red_scores.items()]
        red_scores_list.sort(key=lambda x: x[1], reverse=True)
        
        # 分成高、中、低三个分数段
        total_reds = len(red_scores_list)
        high_tier = red_scores_list[:total_reds//3]
        mid_tier = red_scores_list[total_reds//3:2*total_reds//3]
        low_tier = red_scores_list[2*total_reds//3:]
        
        # 蓝球分层
        blue_scores_list = [(num, score) for num, score in self.blue_scores.items()]
        blue_scores_list.sort(key=lambda x: x[1], reverse=True)
        blue_high = blue_scores_list[:6]
        blue_low = blue_scores_list[6:]
        
        # 定义组合模式：(高分数量, 中分数量, 低分数量)
        score_patterns = [
            (3, 2, 0),  # 高分3个，中分2个
            (2, 3, 0),  # 高分2个，中分3个
            (2, 2, 1),  # 高分2个，中分2个，低分1个
            (1, 3, 1),  # 高分1个，中分3个，低分1个
            (4, 1, 0),  # 高分4个，中分1个
            (1, 4, 0),  # 高分1个，中分4个
        ]
        
        used_combinations = set()
        
        for _ in range(num_combinations * 8):
            if len(combinations) >= num_combinations:
                break
                
            high_count, mid_count, low_count = random.choice(score_patterns)
            
            try:
                selected_reds = []
                if high_count > 0:
                    selected_reds.extend(random.sample([n for n, _ in high_tier], 
                                                     min(high_count, len(high_tier))))
                if mid_count > 0:
                    selected_reds.extend(random.sample([n for n, _ in mid_tier], 
                                                     min(mid_count, len(mid_tier))))
                if low_count > 0:
                    selected_reds.extend(random.sample([n for n, _ in low_tier], 
                                                     min(low_count, len(low_tier))))
                
                # 蓝球：1个高分 + 1个低分，或2个高分，或2个低分
                blue_pattern = random.choice([(1, 1), (2, 0), (0, 2)])
                selected_blues = []
                if blue_pattern[0] > 0:
                    selected_blues.extend(random.sample([n for n, _ in blue_high], 
                                                      min(blue_pattern[0], len(blue_high))))
                if blue_pattern[1] > 0:
                    selected_blues.extend(random.sample([n for n, _ in blue_low], 
                                                      min(blue_pattern[1], len(blue_low))))
                
                if len(selected_reds) == 5 and len(selected_blues) == 2:
                    combo_key = (tuple(sorted(selected_reds)), tuple(sorted(selected_blues)))
                    if combo_key not in used_combinations:
                        combinations.append({
                            'red': sorted(selected_reds),
                            'blue': sorted(selected_blues),
                            'strategy': '分数聚类'
                        })
                        used_combinations.add(combo_key)
                        
            except (ValueError, IndexError):
                continue
                
        return combinations
    
    def strategy_3_diversity_maximization(self, num_combinations: int = 12) -> List[Dict]:
        """
        策略3: 多样性最大化
        确保组合间差异最大化
        """
        combinations = []
        
        # 获取候选池
        red_candidates = sorted(self.red_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        blue_candidates = sorted(self.blue_scores.items(), key=lambda x: x[1], reverse=True)[:8]
        
        red_pool = [num for num, _ in red_candidates]
        blue_pool = [num for num, _ in blue_candidates]
        
        # 生成大量候选组合
        candidate_combinations = []
        for _ in range(num_combinations * 50):
            try:
                red_combo = sorted(random.sample(red_pool, 5))
                blue_combo = sorted(random.sample(blue_pool, 2))
                candidate_combinations.append({
                    'red': red_combo,
                    'blue': blue_combo,
                    'score': sum(self.red_scores.get(r, 0) for r in red_combo) + 
                            sum(self.blue_scores.get(b, 0) for b in blue_combo)
                })
            except ValueError:
                continue
        
        # 去重
        unique_combinations = []
        seen_combos = set()
        for combo in candidate_combinations:
            combo_key = (tuple(combo['red']), tuple(combo['blue']))
            if combo_key not in seen_combos:
                unique_combinations.append(combo)
                seen_combos.add(combo_key)
        
        # 多样性选择算法
        if not unique_combinations:
            return []
            
        # 选择第一个最高分组合
        unique_combinations.sort(key=lambda x: x['score'], reverse=True)
        selected = [unique_combinations[0]]
        remaining = unique_combinations[1:]
        
        # 迭代选择最大多样性组合
        while len(selected) < num_combinations and remaining:
            best_candidate = None
            max_diversity_score = -1
            
            for candidate in remaining:
                # 计算与已选组合的多样性分数
                diversity_score = 0
                for selected_combo in selected:
                    # 红球差异
                    red_diff = len(set(candidate['red']) - set(selected_combo['red']))
                    # 蓝球差异
                    blue_diff = len(set(candidate['blue']) - set(selected_combo['blue']))
                    diversity_score += red_diff + blue_diff * 0.5
                
                # 综合考虑多样性和分数
                final_score = diversity_score * 0.7 + candidate['score'] * 0.0001
                
                if final_score > max_diversity_score:
                    max_diversity_score = final_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        # 转换格式
        for combo in selected:
            combinations.append({
                'red': combo['red'],
                'blue': combo['blue'],
                'strategy': '多样性最大化'
            })
        
        return combinations
    
    def generate_improved_combinations(self, num_combinations: int = 12) -> List[Dict]:
        """
        生成改进的组合，融合多种策略
        """
        all_combinations = []
        
        # 使用各种策略生成组合
        strategy1_combos = self.strategy_1_balanced_zone_selection(num_combinations // 3 + 1)
        strategy2_combos = self.strategy_2_score_clustering(num_combinations // 3 + 1)
        strategy3_combos = self.strategy_3_diversity_maximization(num_combinations // 3 + 1)
        
        # 合并所有策略的结果
        all_combinations.extend(strategy1_combos)
        all_combinations.extend(strategy2_combos)
        all_combinations.extend(strategy3_combos)
        
        # 去重
        unique_combinations = []
        seen_combos = set()
        for combo in all_combinations:
            combo_key = (tuple(combo['red']), tuple(combo['blue']))
            if combo_key not in seen_combos:
                unique_combinations.append(combo)
                seen_combos.add(combo_key)
        
        # 计算综合评分并排序
        for combo in unique_combinations:
            combo['score'] = (sum(self.red_scores.get(r, 0) for r in combo['red']) + 
                            sum(self.blue_scores.get(b, 0) for b in combo['blue']))
        
        # 按分数排序，但保持多样性
        unique_combinations.sort(key=lambda x: x['score'], reverse=True)
        
        # 最终多样性筛选
        final_combinations = []
        for combo in unique_combinations:
            if len(final_combinations) >= num_combinations:
                break
                
            # 检查与已选组合的多样性
            is_diverse = True
            for selected in final_combinations:
                common_reds = len(set(combo['red']) & set(selected['red']))
                if common_reds > 2:  # 最多2个相同红球
                    is_diverse = False
                    break
            
            if is_diverse:
                final_combinations.append(combo)
        
        return final_combinations[:num_combinations]


def format_combinations_output(combinations: List[Dict]) -> List[str]:
    """格式化输出组合"""
    output_lines = [f"改进推荐组合 (Top {len(combinations)}):"]
    
    for i, combo in enumerate(combinations):
        red_str = ' '.join(f'{n:02d}' for n in combo['red'])
        blue_str = ' '.join(f'{n:02d}' for n in combo['blue'])
        strategy = combo.get('strategy', '未知策略')
        score = combo.get('score', 0)
        
        output_lines.append(
            f"  注 {i+1}: 红球 [{red_str}] 蓝球 [{blue_str}] "
            f"(评分: {score:.2f}) - {strategy}"
        )
    
    return output_lines


# 示例使用
if __name__ == "__main__":
    # 模拟分数数据
    red_scores = {i: random.uniform(50, 100) for i in range(1, 36)}
    blue_scores = {i: random.uniform(50, 100) for i in range(1, 13)}
    
    generator = ImprovedCombinationGenerator(red_scores, blue_scores)
    improved_combos = generator.generate_improved_combinations(12)
    
    output_lines = format_combinations_output(improved_combos)
    for line in output_lines:
        print(line) 