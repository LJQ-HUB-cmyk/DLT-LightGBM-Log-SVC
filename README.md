# DLT智能预测分析系统

[![GitHub](https://img.shields.io/badge/GitHub-DLT--LightGBM--Log--SVC-blue?logo=github)](https://github.com/LJQ-HUB-cmyk/DLT-LightGBM-Log-SVC)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> 基于机器学习、统计分析和数据挖掘技术的大乐透彩票智能预测系统

## 🎯 项目简介

本项目是一个专门针对**中国体彩大乐透**的智能预测分析系统，采用多种先进算法相结合的方式：

- **LightGBM机器学习**：预测单个号码出现概率
- **统计分析**：频率、遗漏、周期性分析  
- **关联规则挖掘**：Apriori算法挖掘号码组合模式
- **特征工程**：多维度特征构建和交互分析
- **自动化运行**：GitHub Actions定时执行

### 大乐透规则
- **红球**：从1-35中选择5个号码
- **蓝球**：从1-12中选择2个号码  
- **奖级**：共9个奖级，一等奖需5+2全中

## 🚀 核心功能

### 📊 数据处理模块 (`dlt_data_processor.py`)
- 🌐 自动获取最新大乐透开奖数据
- 🔄 智能数据清洗、验证和格式化
- 📁 生成标准化的历史数据文件

### 🧠 智能分析引擎 (`dlt_analyzer.py`)
- 🔮 **机器学习预测**：LightGBM分类器预测号码概率
- 📈 **统计分析**：频率、遗漏、趋势分析
- 🔗 **关联规则挖掘**：发现号码组合规律
- ⚡ **特征工程**：和值、跨度、奇偶、区间分布等多维特征
- 🎯 **智能推荐**：生成单式精选和复式参考

### 💰 验证计算模块 (`dlt_bonus_calculation.py`)
- ✅ 自动验证历史预测准确性
- 💎 计算9个奖级的中奖情况和奖金
- 📊 生成详细的验证报告

### 🤖 自动化运行
- ⏰ **定时执行**：每周一、三、六早上7点自动运行
- 🔄 **数据更新**：自动获取最新开奖数据
- 📝 **报告生成**：自动生成分析报告和推荐

## 📋 奖级体系

| 奖级 | 中奖条件 | 奖金金额 |
|------|----------|----------|
| 一等奖 | 5+2 | 约1000万元 |
| 二等奖 | 5+1 | 约50万元 |
| 三等奖 | 5+0 | 1万元 |
| 四等奖 | 4+2 | 3000元 |
| 五等奖 | 4+1、3+2 | 300元 |
| 六等奖 | 4+0、3+1、2+2 | 200元 |
| 七等奖 | 3+0、2+1、1+2、0+2 | 100元 |
| 八等奖 | 2+0、1+1、0+1 | 15元 |
| 九等奖 | 1+0、0+0 | 5元 |

## 🛠️ 安装和使用

### 环境要求
- Python 3.8+
- 2GB+ 内存
- 稳定的网络连接

### 快速开始

1. **克隆项目**
```bash
git clone https://github.com/LJQ-HUB-cmyk/DLT-LightGBM-Log-SVC.git
cd DLT-LightGBM-Log-SVC
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **运行分析**
```bash
# 获取最新数据
python dlt_data_processor.py

# 生成预测分析
python dlt_analyzer.py

# 验证历史预测（可选）
python dlt_bonus_calculation.py
```

### 输出示例

**单式推荐**：
```
注 1: 红球 [03 15 22 28 35] 蓝球 [05 09]
注 2: 红球 [01 12 19 26 33] 蓝球 [03 11]
```

**复式参考**：
```
红球 (Top 8): 03 12 15 19 22 26 28 35
蓝球 (Top 4): 03 05 09 11
```

## 📚 项目同步到GitHub的详细步骤

### 方法一：首次上传完整项目

如果您是项目的创建者，需要将本地项目同步到GitHub仓库：

#### 1. 初始化本地Git仓库
```bash
# 在项目根目录下执行
git init
```

#### 2. 添加远程仓库地址
```bash
git remote add origin https://github.com/LJQ-HUB-cmyk/DLT-LightGBM-Log-SVC.git
```

#### 3. 添加所有文件到暂存区
```bash
git add .
```

#### 4. 提交到本地仓库
```bash
git commit -m "🎉 初始提交: 大乐透智能预测分析系统"
```

#### 5. 推送到GitHub
```bash
# 首次推送，建立追踪关系
git push -u origin main
```

### 方法二：克隆现有仓库并更新

如果仓库已存在且您有权限：

#### 1. 克隆仓库
```bash
git clone https://github.com/LJQ-HUB-cmyk/DLT-LightGBM-Log-SVC.git
cd DLT-LightGBM-Log-SVC
```

#### 2. 复制项目文件到克隆的目录

#### 3. 添加更改并提交
```bash
git add .
git commit -m "✨ 添加大乐透预测系统核心功能"
git push origin main
```

### 方法三：日常更新操作

项目运行后的日常更新：

```bash
# 1. 查看文件状态
git status

# 2. 添加修改的文件
git add .

# 3. 提交更改
git commit -m "📊 优化滑动窗口逻辑"

# 4. 推送到远程仓库
git push origin main
```

### 🔧 常见Git命令

```bash
# 查看提交历史
git log --oneline

# 查看远程仓库信息
git remote -v

# 拉取最新更改
git pull origin main

# 查看分支
git branch -a

# 创建新分支
git checkout -b feature/new-algorithm
```

### ⚠️ 注意事项

1. **大文件处理**：如果有大于100MB的文件，考虑使用Git LFS
2. **敏感信息**：确保不要提交包含API密钥或个人信息的文件
3. **.gitignore**：添加不需要版本控制的文件类型
4. **分支管理**：重要更新建议先在分支中开发测试

## 🔄 自动化工作流

项目配置了GitHub Actions自动化工作流，会在以下时间自动运行：

- ⏰ **自动运行时间**：每周一、三、六早上7点（北京时间）
- 🔄 **执行流程**：
  1. 获取最新大乐透开奖数据
  2. 运行预测验证计算
  3. 执行智能分析生成预测
  4. 自动提交结果到仓库

- 🎮 **手动触发**：也可以在GitHub仓库的Actions页面手动触发运行

## 📊 技术架构

```
大乐透预测系统
├── 数据层
│   ├── 网络数据获取 (17500.cn)
│   ├── 数据清洗验证
│   └── CSV文件存储
├── 分析层  
│   ├── 特征工程
│   ├── 统计分析
│   ├── 机器学习 (LightGBM)
│   └── 关联规则挖掘 (Apriori)
├── 预测层
│   ├── 号码评分系统
│   ├── 组合生成算法
│   └── 多样性控制
└── 验证层
    ├── 历史回测
    ├── 中奖计算
    └── 报告生成
```

## 🏆 算法优势

- ✅ **多维度分析**：统计+机器学习+数据挖掘三重保障
- ✅ **智能特征工程**：和值、跨度、奇偶、区间等多种特征
- ✅ **参数自动优化**：Optuna框架自动寻找最优参数
- ✅ **历史验证**：回测功能验证算法有效性
- ✅ **自动化运行**：无人值守定时执行

## 📁 项目文件结构

```
DLT-LightGBM-Log-SVC/
├── dlt_data_processor.py      # 数据获取处理模块
├── dlt_analyzer.py            # 核心分析预测引擎  
├── dlt_bonus_calculation.py   # 奖金计算验证模块
├── requirements.txt           # Python依赖包
├── README.md                  # 项目说明文档
├── .github/
│   └── workflows/
│       └── daily-dlt-analysis.yml  # 自动化工作流
├── daletou.csv               # 历史开奖数据
├── latest_dlt_analysis.txt   # 最新分析报告
└── latest_dlt_calculation.txt # 最新验证结果
```

## 🎮 使用指南

### 本地运行
1. 确保网络连接正常
2. 运行数据处理器获取最新数据
3. 执行分析器生成预测推荐
4. 可选运行验证器查看历史效果

### 云端自动化
- 项目会自动在设定时间运行
- 结果会自动更新到GitHub仓库
- 可在仓库中查看最新的分析报告

## ⚠️ 免责声明

- 🎲 **仅供研究**：本项目仅用于技术研究和学习目的
- 🎯 **理性购彩**：彩票具有随机性，请理性投注，量力而行
- 📊 **数据依赖**：预测效果依赖于历史数据的质量和完整性
- 💰 **风险提示**：任何投注都存在风险，请谨慎决策

## 🔗 相关链接

- 📦 [GitHub仓库](https://github.com/LJQ-HUB-cmyk/DLT-LightGBM-Log-SVC)
- 🎯 [大乐透官网](http://www.cwl.gov.cn/)
- 📊 [数据来源](https://www.17500.cn/)

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

**⭐ 如果此项目对您有帮助，请给个Star支持！**

> 💡 **技术交流**：欢迎提Issue讨论算法改进和功能建议 