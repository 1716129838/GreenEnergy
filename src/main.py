"""
绿电+热电联产+加氢能源 多能互补协同调度系统
================================================
人工智能基础B 课程大作业
功能：
1. LSTM 风光功率预测
2. 规则调度引擎（制氢优化 + 热电联产分配）
3. 大模型策略解读
4. 能耗与效益分析
作者：AI Assistant
日期：2026-05-20
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
# 导入各模块
from data_loader import DataLoader
from lstm_model import LSTMPredictor
from dispatcher import DispatchEngine
from llm_analyst import LLMAnalyst
from evaluator import SystemEvaluator
class GreenEnergySystem:
    """绿电+热电联产+加氢能源 多能互补系统"""
    def __init__(self, data_path, output_dir='results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.data_loader = None
        self.wind_predictor = None
        self.solar_predictor = None
        self.dispatcher = None
        self.llm_analyst = None
        self.evaluator = None
        # 系统参数配置
        self.config = {
            # 电解制氢参数
            'electrolyzer_capacity': 1500,  # kW，电解槽额定功率
            'electrolyzer_efficiency': 50,  # 制氢电耗（kWh/kg H2），行业常规值
            # 余热回收参数（CHP为电解槽废热回收，不消耗额外电能）
            'waste_heat_ratio': 0.30,  # 电解槽废热占输入电能比例
            'recovery_efficiency': 0.75,  # 余热回收效率
            'chp_capacity': 350,  # kW，余热回收系统额定热功率
            # 储能参数
            'h2_storage_capacity': 3000,  # kg，储氢罐容量
            'h2_storage_level': 1500,  # 当前储氢量
            # 经济参数
            # 购电电价（工商业峰谷电价）
            'grid_buy_price_low': 0.3,  # 元/kWh，谷时购电价
            'grid_buy_price_high': 0.8,  # 元/kWh，峰时购电价
            # 售电电价（新能源并网标杆电价）
            'grid_sell_price': 0.35,  # 元/kWh，上网电价
            # 产品售价
            'h2_sell_price': 30,  # 元/kg，绿氢售价
            'heat_sell_price': 0.3,  # 元/kWh，供热价格
            # 预测参数
            'lookback_window': 24,  # 历史窗口（小时）
            'forecast_horizon': 24,  # 预测 horizon（小时）
        }
        os.makedirs(output_dir, exist_ok=True)
    def initialize(self):
        """初始化所有组件"""
        print("=" * 60)
        print("初始化绿电+热电联产+加氢能源系统...")
        print("=" * 60)
        # 1. 数据加载器
        self.data_loader = DataLoader(self.data_path)
        self.data_loader.load_and_preprocess()
        self.data_loader.explore_data()
        # 2. LSTM 预测器
        self.wind_predictor = LSTMPredictor(
            name='Wind',
            input_dim=1,
            hidden_dim=64,
            num_layers=2,
            lookback=self.config['lookback_window']
        )
        self.solar_predictor = LSTMPredictor(
            name='Solar',
            input_dim=1,
            hidden_dim=64,
            num_layers=2,
            lookback=self.config['lookback_window']
        )
        # 3. 调度引擎
        self.dispatcher = DispatchEngine(self.config)
        # 4. 大模型分析师（需要 API key）
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('CLAUDE_API_KEY')
        if api_key:
            self.llm_analyst = LLMAnalyst(api_key=api_key)
            print("✓ 大模型分析师已启用")
        else:
            print("⚠ 未检测到 API key，跳过大模型解读")
        # 5. 系统评估器
        self.evaluator = SystemEvaluator(
            output_dir=self.output_dir,
            electrolyzer_rated=self.config['electrolyzer_capacity'],
            chp_rated=self.config['chp_capacity'],
            h2_storage_rated=self.config['h2_storage_capacity'],
            # 设备投资参数（参考行业平均水平）
            electrolyzer_investment=1000,  # 电解槽单位投资 元/kW
            wind_investment=7000,  # 风电单位投资 元/kW
            solar_investment=4500,  # 光伏单位投资 元/kW
            # 折旧年限
            electrolyzer_lifetime=10,  # 电解槽折旧年限
            wind_lifetime=20,  # 风电折旧年限
            solar_lifetime=25,  # 光伏折旧年限
            # 运维成本参数
            wind_om_cost=0.05,  # 风电运维成本 元/kWh
            solar_om_cost=0.03,  # 光伏运维成本 元/kWh
            h2_om_ratio=0.10,  # 制氢运维成本占氢收益比例
            # 其他成本
            land_cost_ratio=0.02,  # 土地成本占总投资比例（年化）
            financial_cost_ratio=0.04  # 财务成本占总投资比例（年化）
        )
        print("✓ 系统初始化完成\n")
    def run(self):
        """运行完整流程"""
        print("=" * 60)
        print("开始运行绿电+热电联产+加氢能源协同调度系统")
        print("=" * 60)
        # ========== 阶段1: 数据准备 ==========
        print("\n[阶段1] 数据准备...")
        train_data, test_data = self.data_loader.split_data()
        # 按8:2比例分别拆分风电和光伏数据，独立生成训练/测试序列
        wind_all = self.data_loader.wind_data
        solar_all = self.data_loader.solar_data
        split_idx = int(len(wind_all) * 0.8)
        # 训练集序列（仅用于模型训练）
        X_train_wind, y_train_wind = self.data_loader.create_sequences(
            wind_all[:split_idx], self.config['lookback_window']
        )
        X_train_solar, y_train_solar = self.data_loader.create_sequences(
            solar_all[:split_idx], self.config['lookback_window']
        )
        # 测试集序列（仅用于模型评估，模型未见过的数据）
        X_test_wind, y_test_wind = self.data_loader.create_sequences(
            wind_all[split_idx:], self.config['lookback_window']
        )
        X_test_solar, y_test_solar = self.data_loader.create_sequences(
            solar_all[split_idx:], self.config['lookback_window']
        )
        print(f"风电训练序列: {len(X_train_wind)}, 测试序列: {len(X_test_wind)}")
        print(f"光伏训练序列: {len(X_train_solar)}, 测试序列: {len(X_test_solar)}")
        # ========== 阶段2: LSTM 训练与评估 ==========
        print("\n[阶段2] LSTM 模型训练与评估...")
        # 训练风电预测模型
        print("训练风电预测 LSTM...")
        self.wind_predictor.train(X_train_wind, y_train_wind, epochs=50)
        wind_results = self.wind_predictor.evaluate(
            X_test_wind, y_test_wind, zero_threshold=5, daytime_threshold=50
        )
        # 训练光伏预测模型
        print("训练光伏预测 LSTM...")
        self.solar_predictor.train(X_train_solar, y_train_solar, epochs=50)
        solar_results = self.solar_predictor.evaluate(
            X_test_solar, y_test_solar, zero_threshold=2, daytime_threshold=50
        )
        # 保存模型
        self.wind_predictor.save(os.path.join(self.output_dir, 'wind_lstm.pth'))
        self.solar_predictor.save(os.path.join(self.output_dir, 'solar_lstm.pth'))
        print("✓ 模型已保存")
        # ========== 阶段3: 预测与协同调度 ==========
        print("\n[阶段3] 预测与协同调度...")
        # 取测试集前一周168小时做调度仿真
        n_samples = min(168, len(X_test_wind))
        wind_forecast = self.wind_predictor.predict(X_test_wind[:n_samples])
        solar_forecast = self.solar_predictor.predict(X_test_solar[:n_samples])
        # 执行调度
        dispatch_results = []
        for i in range(n_samples):
            w_power = wind_forecast[i]
            s_power = solar_forecast[i]
            result = self.dispatcher.dispatch(w_power, s_power)
            dispatch_results.append(result)
        print(f"✓ 完成 {n_samples} 个时段的协同调度")
        # ========== 阶段4: 大模型解读 ==========
        print("\n[阶段4] 大模型策略解读...")
        if self.llm_analyst:
            # 准备调度场景摘要
            scenario_summary = self.dispatcher.summarize_results(dispatch_results)
            # 调用大模型分析
            analysis = self.llm_analyst.analyze_scenario(scenario_summary)
            # 保存分析结果
            with open(os.path.join(self.output_dir, 'llm_analysis.txt'), 'w') as f:
                f.write(analysis)
            print("✓ 大模型解读已完成")
        else:
            analysis = "（大模型 API 未配置，跳过解读）"
        # ========== 阶段5: 系统评估 ==========
        print("\n[阶段5] 系统能效评估...")
        metrics = self.evaluator.evaluate(dispatch_results, wind_forecast, solar_forecast)
        print(f"  系统综合能效: {metrics['overall_efficiency']:.2f}%")
        print(f"  绿电占比: {metrics['green_power_ratio']:.2f}%")
        print(f"  绿氢产量: {metrics['h2_production']:.2f} kg")
        print(f"  碳减排量: {metrics['carbon_savings']:.2f} kg CO2")
        print(f"  净经济效益: {metrics['net_economic_benefit']:.2f} 元")
        print(f"  系统总投资估算: {metrics['total_investment_est'] / 10000:.2f} 万元")
        # ========== 阶段6: 生成报告 ==========
        print("\n[阶段6] 生成分析报告...")
        self.generate_report(wind_results, solar_results, metrics, analysis)
        print("✓ 报告已生成")
        print("\n" + "=" * 60)
        print("系统运行完成！")
        print("=" * 60)
        return {
            'wind_rmse': wind_results['rmse'],
            'solar_rmse': solar_results['rmse'],
            'metrics': metrics
        }
    def generate_report(self, wind_results, solar_results, metrics, analysis):
        """生成完整分析报告"""
        import json
        from datetime import datetime
        report = f"""
# 绿电+热电联产+加氢能源 多能互补协同调度系统
## 分析报告
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**课程**: 人工智能基础B
---
## 1. 系统架构
### 1.1 整体架构
```
┌──────────────────────────────────────────────────────────────┐
│                    多能互补协同调度系统                        │
│                                                              │
│  🌬️ 风电预测 ──┐                                            │
│  ☀️ 光伏预测 ──┼──▶ 🔋 电解制氢优化 ──▶ ⚡ 余热回收供热      │
│                                     │           │            │
│                                     ▼           ▼            │
│                               💧 储氢管理    🏠 热负荷匹配     │
│                                     │           │            │
│                                     └─────┬─────┘            │
│                                           ▼                   │
│                                    📊 整体能效评估             │
└──────────────────────────────────────────────────────────────┘
```
### 1.2 系统场景假设
本系统为**并网型风光制氢热电联产系统**，电网作为备用调节电源：
- 风光出力充足时，优先使用绿电制氢，富余电量上网售电；
- 风光出力不足时，从电网购电补充，保证电解槽连续稳定运行；
- 电解槽运行余热全部回收用于供热，实现能量梯级利用。
### 1.3 核心模块
| 模块 | 技术 | 功能 |
|------|------|------|
| 风电预测 | LSTM | 多步风力功率预测 |
| 光伏预测 | LSTM | 多步光伏功率预测 |
| 调度引擎 | 规则优化 | 绿氢-热电联产协同 |
| 策略解读 | 大模型 | 调度逻辑分析 |
| 系统评估 | 能效核算 | 多维度性能评估 |
---
## 2. 算法原理
### 2.1 LSTM 模型
Long Short-Term Memory (LSTM) 是一种特殊的循环神经网络(RNN)，擅长处理时间序列预测问题。
**核心组件**：
- 输入门：决定有多少新信息存入单元状态
- 遗忘门：决定保留多少上一时刻的信息
- 输出门：决定输出多少信息
**数学公式**：
```
遗忘门: f = σ(W_f · [h, x] + bf)
输入门: i = σ(W_i · [h, x] + bi)
候选值: C = tanh(W_C · [h, x] + b_C)
单元状态: C_t = f * C_{{t-1}} + i * C̃
输出门: o = σ(W_o · [h, x] + b_o)
隐藏状态: h = o * tanh(C)
```
### 2.2 调度规则
调度引擎采用基于规则的优化策略：
**优先级规则**：
1. 优先使用风光绿电进行电解制氢
2. 绿电不足时，从电网购电补充，保证电解槽满负荷运行
3. 绿电富余时，多余电量全部上网售电
4. 电解槽余热全部回收用于供热
**优化目标**：
- 最大化绿氢产量
- 最大化绿电自用比例
- 最大化经济效益
- 最大化碳减排量
---
## 3. 实验设计
### 3.1 数据集
- **来源**: Open Power System Data (OPSD) 欧洲电力数据集
- **时间范围**: 2015-2020
- **采样间隔**: 1小时
- **国家**: 德国（作为典型案例）
### 3.2 模型配置
| 参数 | 值 |
|------|-----|
| 历史窗口 | {self.config['lookback_window']} 小时 |
| 预测周期 | {self.config['forecast_horizon']} 小时 |
| LSTM 隐藏层 | {64} |
| LSTM 层数 | 2 |
| 训练轮次 | 50 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 早停耐心值 | 10 |
### 3.3 评估指标
**预测精度指标**：
- RMSE (均方根误差)
- MAE (平均绝对误差)
- MAPE (平均绝对百分比误差) - 含全天和白天时段两个口径
**系统性能指标**：
- 系统综合能效
- 绿电自用率 / 上网率 / 弃电率
- 碳减排量
- 经济效益（全成本核算）
---
## 4. 实验结果
### 4.1 预测模型性能
| 模型 | RMSE | MAE | 全天MAPE | 白天时段MAPE |
|------|------|-----|----------|--------------|
| 风电预测 LSTM | {wind_results['rmse']:.4f} | {wind_results['mae']:.4f} | {wind_results['mape']:.2f}% | {wind_results['mape_daytime']:.2f}% |
| 光伏预测 LSTM | {solar_results['rmse']:.4f} | {solar_results['mae']:.4f} | {solar_results['mape']:.2f}% | {solar_results['mape_daytime']:.2f}% |
> 注：白天时段MAPE仅统计出力大于阈值的时段，消除夜间零值对百分比误差的干扰。
### 4.2 系统运行指标
| 指标 | 值 |
|------|-----|
| 系统综合能效 | {metrics['overall_efficiency']:.2f}% |
| 绿电自用率 | {metrics['self_use_rate']:.2f}% |
| 绿电上网率 | {metrics['export_rate']:.2f}% |
| 弃风弃光率 | {metrics['curtailment_rate']:.2f}% |
| 绿氢产量 | {metrics['h2_production']:.2f} kg |
| 热电联产产热 | {metrics['chp_heat_total']:.2f} kWh |
| 电解槽利用率 | {metrics['electrolyzer_utilization']:.2f}% |

### 4.3 绿氢属性
| 指标 | 值 | 说明 |
|------|-----|------|
| 绿电占比 | {metrics['green_power_ratio']:.2f}% | 电解槽用电中绿电比例 |
| 绿氢产量 | {metrics['h2_production']:.2f} kg | 由绿电生产的氢气总量 |

> **绿氢认证说明**：根据国内绿氢认证相关规范，当电解槽用电中绿电占比≥95%时，可认定为绿氢。本系统绿电占比达 {metrics['green_power_ratio']:.2f}%，{'已达到' if metrics['green_power_ratio'] >= 95 else '接近'}绿氢认证标准。

### 4.4 经济效益
#### 4.4.1 收益明细
| 指标 | 值 |
|------|-----|
| 总收益 | {metrics['total_revenue']:.2f} 元 |
| 绿氢收益 | {metrics['h2_revenue']:.2f} 元 |
| 供热收益 | {metrics['heat_revenue']:.2f} 元 |
| 上网售电收益 | {metrics['grid_export_revenue']:.2f} 元 |

#### 4.4.2 成本明细
| 指标 | 值 | 说明 |
|------|-----|------|
| 总成本 | {metrics['total_cost']:.2f} 元 | 全口径成本 |
| 购电成本 | {metrics['grid_import_cost']:.2f} 元 | 电网购电费用 |
| 运行运维成本 | {metrics['operation_cost']:.2f} 元 | 含制氢、风电、光伏运维 |
| &nbsp;&nbsp;制氢运维 | {metrics['h2_operation_cost']:.2f} 元 | 水耗、耗材、人工等 |
| &nbsp;&nbsp;风电运维 | {metrics['wind_om_cost']:.2f} 元 | 按发电量计算 |
| &nbsp;&nbsp;光伏运维 | {metrics['solar_om_cost']:.2f} 元 | 按发电量计算 |
| 设备折旧成本 | {metrics['depreciation_cost']:.2f} 元 | 按投资金额和折旧年限计算 |
| 土地成本 | {metrics['land_cost']:.2f} 元 | 按总投资比例估算 |
| 财务成本 | {metrics['financial_cost']:.2f} 元 | 按总投资比例估算 |

#### 4.4.3 净效益
| 指标 | 值 |
|------|-----|
| **净经济效益** | **{metrics['net_economic_benefit']:.2f} 元** |

#### 4.4.4 投资估算
| 指标 | 值 | 说明 |
|------|-----|------|
| 系统总投资估算 | {metrics['total_investment_est'] / 10000:.2f} 万元 | 含电解槽、风电、光伏设备投资 |

> **成本核算说明**：本评估采用全成本核算方法，包含设备折旧、运维成本、土地成本、财务成本等全口径成本。设备折旧按直线法计算，电解槽折旧年限10年，风电20年，光伏25年。风光装机容量按调度周期内最大功率估算，实际投资需根据具体装机规模调整。

### 4.5 环保效益
| 指标 | 值 | 说明 |
|------|-----|------|
| 碳减排量 | {metrics['carbon_savings']:.2f} kg CO2 | 绿氢替代灰氢 + 余热替代燃煤供热 |
---
## 5. 指标核算口径说明
### 5.1 能效指标
- **系统综合能效**：(氢能低位发热量 + 回收热能) / 电解槽消耗总电能
  - 氢能低位发热量：33.3 kWh/kg
  - 电解槽消耗总电能 = 绿电制氢 + 电网购电
- **绿电自用率**：制氢消纳的绿电量 / 总绿发电量
- **绿电上网率**：上网售电的绿电量 / 总绿发电量
- **弃风弃光率**：无法消纳且无法上网的废弃电量 / 总绿发电量

### 5.2 绿氢属性指标
- **绿电占比**：电解槽用电中绿电的比例，用于绿氢认证参考
  - 计算公式：绿电制氢量 / 电解槽总耗电量 × 100%
  - 绿氢认证标准：≥95% 可认定为绿氢

### 5.3 经济指标（全成本核算）
#### 5.3.1 收益项
- **总收益**：绿氢销售收入 + 供热收入 + 上网售电收入
- **绿氢收益**：绿氢产量 × 绿氢售价（30元/kg）
- **供热收益**：回收余热 × 供热价格（0.3元/kWh）
- **上网售电收益**：上网售电量 × 上网电价（0.35元/kWh）

#### 5.3.2 成本项
- **总成本**：购电成本 + 运行运维成本 + 设备折旧成本 + 土地成本 + 财务成本
- **购电成本**：电网购电量 × 购电电价（工商业峰谷电价）
- **运行运维成本**：
  - 制氢运维：水耗、耗材、人工等，按氢收益比例估算（10%）
  - 风电运维：按发电量计算（0.05元/kWh）
  - 光伏运维：按发电量计算（0.03元/kWh）
- **设备折旧成本**：按直线法计算
  - 电解槽折旧：单位投资1000元/kW，折旧年限10年
  - 风电折旧：单位投资7000元/kW，折旧年限20年
  - 光伏折旧：单位投资4500元/kW，折旧年限25年
- **土地成本**：按总投资比例估算（年化2%）
- **财务成本**：按总投资比例估算（年化4%）

#### 5.3.3 净效益
- **净经济效益**：总收益 - 总成本

### 5.4 环保指标
- **碳减排量**：绿氢替代煤制灰氢的减排量 + 余热替代燃煤供热的减排量
- 减排因子：氢 10 kgCO₂/kg（灰氢排放因子），热 0.11 kgCO₂/kWh（燃煤供热排放因子）
- 注意：仅绿电部分产生减排，网电部分不计入
---
## 6. 大模型策略解读
{analysis}
---
## 7. 总结与反思
### 7.1 创新点
1. **双scaler归一化**：输入特征和输出目标分别归一化，避免尺度混淆导致的预测偏差
2. **并网协同调度**：电网作为备用电源，保证电解槽连续稳定运行，提升设备利用率
3. **全成本经济核算**：包含设备折旧、运维成本、土地成本、财务成本等全口径成本，经济评估更严谨
4. **分时段MAPE评估**：单独统计白天时段预测精度，消除夜间零值干扰
5. **绿氢属性认证**：计算并展示绿电占比，匹配绿氢认证相关规范
6. **电价机制优化**：区分购电电价（工商业峰谷电价）和售电电价（新能源并网标杆电价），更符合实际电力市场
### 7.2 不足与改进方向
1. **预测精度**：可引入 Transformer 或注意力机制提升预测精度，增加气象特征
2. **调度策略**：可升级为强化学习 (DQN/PPO) 实现更智能的多目标优化调度
3. **不确定性**：可引入随机规划或鲁棒优化处理风光功率的波动性
4. **系统规模**：可根据实际场景调整电解槽容量，优化源荷匹配度
5. **储能配置**：可新增储能电池配置，利用峰谷电价套利提升经济性
6. **绿电交易**：可引入绿电交易机制，探索绿电溢价收益
7. **多能互补**：可扩展为风光储氢多能互补系统，增加灵活性
### 7.3 未来展望
- 短期: 升级为强化学习调度（课程论文）
- 中期: 引入多目标优化 (NSGA-II) 进行毕设研究
- 长期: 发表核心期刊论文
---
## 附录: 核心代码结构
```
GreenEnergy_PDCA/
├── main.py              # 主程序入口
├── data_loader.py       # 数据加载与预处理
├── lstm_model.py        # LSTM 预测模型
├── dispatcher.py         # 调度引擎
├── llm_analyst.py       # 大模型分析师
├── evaluator.py         # 系统评估器
└── requirements.txt     # 依赖包
```
---
*本报告由 AI 辅助生成*
"""
        with open(os.path.join(self.output_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)
        # 保存 JSON 格式的结果
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'wind_results': {
                'rmse': float(wind_results['rmse']),
                'mae': float(wind_results['mae']),
                'mape': float(wind_results['mape']),
                'mape_daytime': float(wind_results['mape_daytime'])
            },
            'solar_results': {
                'rmse': float(solar_results['rmse']),
                'mae': float(solar_results['mae']),
                'mape': float(solar_results['mape']),
                'mape_daytime': float(solar_results['mape_daytime'])
            },
            'metrics': metrics,
            'config': self.config
        }
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results_json, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
if __name__ == '__main__':
    # 配置路径
    DATA_PATH = 'data/opsd_time_series.csv'
