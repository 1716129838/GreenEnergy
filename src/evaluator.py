"""
系统评估器模块
评估多能互补系统的整体性能
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免无GUI环境报错
import matplotlib.pyplot as plt
import os
class SystemEvaluator:
    """系统性能评估器"""
    def __init__(self, output_dir='results',
                 electrolyzer_rated=500,
                 chp_rated=300,
                 h2_storage_rated=1000,
                 # 设备投资参数（万元）
                 electrolyzer_investment=None,  # 电解槽单位投资 元/kW，默认1000
                 wind_investment=None,  # 风电单位投资 元/kW，默认7000
                 solar_investment=None,  # 光伏单位投资 元/kW，默认4500
                 # 折旧年限（年）
                 electrolyzer_lifetime=10,
                 wind_lifetime=20,
                 solar_lifetime=25,
                 # 运维成本参数
                 wind_om_cost=0.05,  # 风电运维成本 元/kWh
                 solar_om_cost=0.03,  # 光伏运维成本 元/kWh
                 h2_om_ratio=0.10,  # 制氢运维成本占氢收益比例
                 # 其他成本
                 land_cost_ratio=0.02,  # 土地成本占总投资比例（年化）
                 financial_cost_ratio=0.04  # 财务成本占总投资比例（年化）
                 ):
        self.output_dir = output_dir
        self.electrolyzer_rated = electrolyzer_rated
        self.chp_rated = chp_rated
        self.h2_storage_rated = h2_storage_rated

        # 设备投资参数（默认值参考行业平均水平）
        self.electrolyzer_investment = electrolyzer_investment or 1000  # 元/kW
        self.wind_investment = wind_investment or 7000  # 元/kW
        self.solar_investment = solar_investment or 4500  # 元/kW

        # 折旧年限
        self.electrolyzer_lifetime = electrolyzer_lifetime
        self.wind_lifetime = wind_lifetime
        self.solar_lifetime = solar_lifetime

        # 运维成本参数
        self.wind_om_cost = wind_om_cost
        self.solar_om_cost = solar_om_cost
        self.h2_om_ratio = h2_om_ratio

        # 其他成本参数
        self.land_cost_ratio = land_cost_ratio
        self.financial_cost_ratio = financial_cost_ratio

        os.makedirs(output_dir, exist_ok=True)
    def evaluate(self, dispatch_results, wind_forecast, solar_forecast):
        """
        评估系统整体性能（并网场景：多余电量优先上网，超出部分弃电）
        """
        if not dispatch_results:
            return self._default_metrics()
        df = pd.DataFrame(dispatch_results)
        # ========== 1. 能效与电量平衡指标 ==========
        green_total = df['total_green_power'].sum()
        # 绿电自用：仅制氢消耗的绿电部分（green_for_h2）
        # 注意：electrolyzer_power 包含电网补电，不是纯绿电
        if 'green_for_h2' in df.columns:
            green_used = df['green_for_h2'].sum()
        else:
            green_used = df['electrolyzer_power'].sum()
        # 电解槽总耗电 = 绿电制氢 + 电网补电 = 系统总输入电能
        electrolyzer_total_power = df['electrolyzer_power'].sum()
        # 上网售电量
        grid_export_sum = df['grid_export'].sum() if 'grid_export' in df.columns else 0
        # 弃电量 = 总绿电 - 自用 - 上网（并网场景下，无法消纳且无法上网的电量）
        curtailment_sum = df['curtailment'].sum() if 'curtailment' in df.columns else max(green_total - green_used - grid_export_sum, 0)
        # 电网购电量
        grid_import_sum = df['grid_import'].sum() if 'grid_import' in df.columns else 0
        # 比率计算
        self_use_rate = green_used / green_total if green_total > 0 else 0
        export_rate = grid_export_sum / green_total if green_total > 0 else 0
        curtailment_rate = curtailment_sum / green_total if green_total > 0 else 0
        # 系统综合能效 = 有效能量产出 / 电解槽实际消耗总电能
        # 有效能量产出 = 氢气化学能 + 回收余热
        # 电解槽总耗电 = 绿电 + 网电
        h2_energy = df['h2_produced'].sum() * 33.3  # 氢气低位发热量 33.3 kWh/kg
        chp_heat_energy = df['chp_heat'].sum()
        total_output = h2_energy + chp_heat_energy
        total_input = electrolyzer_total_power  # 电解槽总耗电就是系统总输入
        overall_efficiency = total_output / total_input if total_input > 0 else 0
        # ========== 2. 经济指标（全成本核算） ==========
        # 主营收益（直接从调度结果获取，确保与调度逻辑一致）
        h2_revenue = df['h2_revenue'].sum() if 'h2_revenue' in df.columns else df['h2_produced'].sum() * 30
        heat_revenue = df['heat_revenue'].sum() if 'heat_revenue' in df.columns else chp_heat_energy * 0.3
        grid_export_revenue = df['grid_export_revenue'].sum() if 'grid_export_revenue' in df.columns else grid_export_sum * 0.35
        grid_import_cost = df['grid_import_cost'].sum() if 'grid_import_cost' in df.columns else grid_import_sum * 0.5

        # 运行运维成本
        # 1. 制氢运维：水耗、耗材、人工等，按氢收益比例估算
        h2_operation_cost = h2_revenue * self.h2_om_ratio
        # 2. 风电运维：按发电量计算
        wind_om_cost = df['wind_power'].sum() * self.wind_om_cost
        # 3. 光伏运维：按发电量计算
        solar_om_cost = df['solar_power'].sum() * self.solar_om_cost
        # 总运维成本
        operation_cost = h2_operation_cost + wind_om_cost + solar_om_cost

        # 设备折旧成本（按投资金额和折旧年限计算）
        # 估算风光装机容量：按调度周期内的平均功率估算（简化处理）
        # 更准确的方式是由外部传入实际装机容量，这里做简化估算
        total_hours = len(df)
        wind_capacity_est = df['wind_power'].max()  # 估算风电装机
        solar_capacity_est = df['solar_power'].max()  # 估算光伏装机

        # 年化折旧 = 装机容量 × 单位投资 / 折旧年限
        electrolyzer_annual_depreciation = self.electrolyzer_rated * self.electrolyzer_investment / self.electrolyzer_lifetime
        wind_annual_depreciation = wind_capacity_est * self.wind_investment / self.wind_lifetime
        solar_annual_depreciation = solar_capacity_est * self.solar_investment / self.solar_lifetime

        # 调度周期内的折旧 = 年化折旧 × (周期小时数 / 全年小时数)
        hours_per_year = 8760
        period_ratio = total_hours / hours_per_year
        depreciation_cost = (electrolyzer_annual_depreciation + wind_annual_depreciation + solar_annual_depreciation) * period_ratio

        # 其他成本：土地成本、财务成本
        total_investment = (self.electrolyzer_rated * self.electrolyzer_investment +
                           wind_capacity_est * self.wind_investment +
                           solar_capacity_est * self.solar_investment)
        land_cost = total_investment * self.land_cost_ratio * period_ratio
        financial_cost = total_investment * self.financial_cost_ratio * period_ratio

        # 全口径收支
        total_revenue = h2_revenue + heat_revenue + grid_export_revenue
        total_cost = grid_import_cost + operation_cost + depreciation_cost + land_cost + financial_cost
        net_benefit = total_revenue - total_cost

        # ========== 2.5 绿氢属性指标 ==========
        # 绿电占比：电解槽用电中绿电的比例
        if 'green_power_ratio' in df.columns:
            green_power_ratio = df['green_power_ratio'].mean() * 100
        elif 'green_for_h2' in df.columns:
            green_power_ratio = df['green_for_h2'].sum() / electrolyzer_total_power * 100 if electrolyzer_total_power > 0 else 0
        else:
            green_power_ratio = 100  # 兼容旧版本，假设全部为绿电
        # ========== 3. 环保指标 ==========
        # 碳减排：绿氢替代灰氢 + 余热替代燃煤供热
        # 注意：仅绿电部分产生减排，网电部分不计入
        h2_carbon_factor = 10  # kgCO2/kg H2（灰氢排放因子）
        heat_carbon_factor = 0.11  # kgCO2/kWh（燃煤供热排放因子）
        if 'green_for_h2' in df.columns:
            green_h2_total = df['green_for_h2'].sum() / 50  # 绿电生产的氢气
            green_heat_total = df['green_for_h2'].sum() * 0.3 * 0.75  # 绿电对应的余热
        else:
            green_h2_total = df['h2_produced'].sum()
            green_heat_total = chp_heat_energy
        h2_carbon_saving = green_h2_total * h2_carbon_factor
        heat_carbon_saving = green_heat_total * heat_carbon_factor
        carbon_savings = h2_carbon_saving + heat_carbon_saving
        # ========== 4. 设备利用率 ==========
        electrolyzer_util = df['electrolyzer_power'].mean() / self.electrolyzer_rated * 100
        chp_util = df['chp_power'].mean() / self.chp_rated * 100 if 'chp_power' in df.columns and self.chp_rated > 0 else 0
        h2_storage_mean = df['h2_storage_level'].mean() if 'h2_storage_level' in df.columns else 0
        h2_utilization = h2_storage_mean / self.h2_storage_rated * 100
        # ========== 5. 封装结果 ==========
        metrics = {
            'overall_efficiency': overall_efficiency * 100,
            'self_use_rate': self_use_rate * 100,
            'export_rate': export_rate * 100,
            'curtailment_rate': curtailment_rate * 100,
            'h2_production': df['h2_produced'].sum(),
            'chp_heat_total': chp_heat_energy,
            'green_power_total': green_total,
            'green_power_used': green_used,
            'grid_import_total': grid_import_sum,
            'grid_export_total': grid_export_sum,
            # 绿氢属性
            'green_power_ratio': green_power_ratio,  # 电解槽用电中绿电占比
            # 收益项
            'total_revenue': total_revenue,
            'h2_revenue': h2_revenue,
            'heat_revenue': heat_revenue,
            'grid_export_revenue': grid_export_revenue,
            # 成本项
            'total_cost': total_cost,
            'grid_import_cost': grid_import_cost,
            'operation_cost': operation_cost,
            'h2_operation_cost': h2_operation_cost,
            'wind_om_cost': wind_om_cost,
            'solar_om_cost': solar_om_cost,
            'depreciation_cost': depreciation_cost,
            'land_cost': land_cost,
            'financial_cost': financial_cost,
            'net_economic_benefit': net_benefit,
            # 投资估算
            'total_investment_est': total_investment,
            # 环保指标
            'carbon_savings': carbon_savings,
            # 设备利用率
            'h2_utilization': h2_utilization,
            'electrolyzer_utilization': electrolyzer_util,
            'chp_utilization': chp_util
        }
        self._print_metrics(metrics)
        self._generate_plots(df, wind_forecast, solar_forecast)
        return metrics
    def _print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "=" * 50)
        print("系统性能评估报告")
        print("=" * 50)

        print(f"【能效指标】")
        print(f"  系统综合能效: {metrics['overall_efficiency']:.2f}%")
        print(f"  绿电自用率: {metrics['self_use_rate']:.2f}%")
        print(f"  绿电上网率: {metrics['export_rate']:.2f}%")
        print(f"  弃风弃光率: {metrics['curtailment_rate']:.2f}%")

        print(f"\n【绿氢属性】")
        print(f"  绿电占比: {metrics['green_power_ratio']:.2f}%")
        print(f"  （电解槽用电中绿电比例，用于绿氢认证参考）")

        print(f"\n【生产指标】")
        print(f"  绿氢总产量: {metrics['h2_production']:.2f} kg")
        print(f"  热电联产产热: {metrics['chp_heat_total']:.2f} kWh")
        print(f"  电解槽利用率: {metrics['electrolyzer_utilization']:.2f}%")

        print(f"\n【电量平衡】")
        print(f"  总绿电量: {metrics['green_power_total']:.2f} kWh")
        print(f"  绿电自用: {metrics['green_power_used']:.2f} kWh")
        print(f"  电网购电: {metrics['grid_import_total']:.2f} kWh")
        print(f"  上网售电: {metrics['grid_export_total']:.2f} kWh")

        print(f"\n【经济指标（全成本核算）】")
        print(f"  总收益: {metrics['total_revenue']:.2f} 元")
        print(f"    绿氢收益: {metrics['h2_revenue']:.2f} 元")
        print(f"    供热收益: {metrics['heat_revenue']:.2f} 元")
        print(f"    上网售电收益: {metrics['grid_export_revenue']:.2f} 元")
        print(f"  总成本: {metrics['total_cost']:.2f} 元")
        print(f"    购电成本: {metrics['grid_import_cost']:.2f} 元")
        print(f"    运行运维成本: {metrics['operation_cost']:.2f} 元")
        print(f"      制氢运维: {metrics['h2_operation_cost']:.2f} 元")
        print(f"      风电运维: {metrics['wind_om_cost']:.2f} 元")
        print(f"      光伏运维: {metrics['solar_om_cost']:.2f} 元")
        print(f"    设备折旧成本: {metrics['depreciation_cost']:.2f} 元")
        print(f"    土地成本: {metrics['land_cost']:.2f} 元")
        print(f"    财务成本: {metrics['financial_cost']:.2f} 元")
        print(f"  净经济效益: {metrics['net_economic_benefit']:.2f} 元")

        print(f"\n【投资估算】")
        print(f"  系统总投资估算: {metrics['total_investment_est'] / 10000:.2f} 万元")
        print(f"  （含电解槽、风电、光伏设备投资）")

        print(f"\n【环保指标】")
        print(f"  碳减排量: {metrics['carbon_savings']:.2f} kg CO2")
        print("=" * 50)
    def _generate_plots(self, df, wind_forecast, solar_forecast):
        """生成可视化图表"""
        wind_arr = np.array(wind_forecast)
        solar_arr = np.array(solar_forecast)
        n = min(len(wind_arr), len(solar_arr), len(df))
        # 图1: 功率调度时序图
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        # 子图1: 风光预测
        ax1 = axes[0]
        hours = range(n)
        ax1.plot(hours, wind_arr[:n], label='Wind', alpha=0.8)
        ax1.plot(hours, solar_arr[:n], label='Solar', alpha=0.8)
        ax1.plot(hours, wind_arr[:n] + solar_arr[:n], label='Total Green', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Power (kW)')
        ax1.set_title('Wind & Solar Power Forecast')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 子图2: 调度功率分配
        ax2 = axes[1]
        electrolyzer = df['electrolyzer_power'].values[:n] if 'electrolyzer_power' in df.columns else np.zeros(n)
        chp = df['chp_power'].values[:n] if 'chp_power' in df.columns else np.zeros(n)
        grid_exp = df['grid_export'].values[:n] if 'grid_export' in df.columns else np.zeros(n)
        ax2.stackplot(hours, electrolyzer, chp, grid_exp,
                      labels=['Electrolyzer', 'CHP', 'Grid Export'],
                      alpha=0.8)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('Power Dispatch Allocation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # 子图3: 储氢水平变化
        ax3 = axes[2]
        h2_storage = df['h2_storage_level'].values[:n] if 'h2_storage_level' in df.columns else np.zeros(n)
        ax3.fill_between(hours, 0, h2_storage, alpha=0.5, label='H2 Storage')
        ax3.axhline(y=self.h2_storage_rated, color='r', linestyle='--', label='Max Capacity')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('H2 (kg)')
        ax3.set_title('Hydrogen Storage Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dispatch_overview.png'), dpi=150)
        print(f"dispatch_overview.png saved")
        # 图2: 经济与碳排放
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
        # 经济收益
        ax_econ = axes2[0]
        h2_rev = df['h2_revenue'].values[:n] if 'h2_revenue' in df.columns else np.zeros(n)
        net_ben = df['net_economic_benefit'].values[:n] if 'net_economic_benefit' in df.columns else np.zeros(n)
        ax_econ.bar(hours, h2_rev, label='H2 Revenue', alpha=0.7)
        ax_econ.bar(hours, np.maximum(net_ben - h2_rev, 0), bottom=h2_rev, label='Other Benefit', alpha=0.7)
        ax_econ.set_xlabel('Time (hours)')
        ax_econ.set_ylabel('Revenue (Yuan)')
        ax_econ.set_title('Economic Benefit per Hour')
        ax_econ.legend()
        ax_econ.grid(True, alpha=0.3)
        # 碳减排
        ax_carbon = axes2[1]
        carbon = df['carbon_savings'].values[:n] if 'carbon_savings' in df.columns else np.zeros(n)
        ax_carbon.fill_between(hours, 0, carbon, alpha=0.5, color='green')
        ax_carbon.set_xlabel('Time (hours)')
        ax_carbon.set_ylabel('Carbon Savings (kg CO2)')
        ax_carbon.set_title('Carbon Emission Reduction')
        ax_carbon.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'economics_carbon.png'), dpi=150)
        print(f"economics_carbon.png saved")
        plt.close('all')
    def _default_metrics(self):
        """默认指标（当无数据时）"""
        return {
            'overall_efficiency': 0,
            'self_use_rate': 0,
            'export_rate': 0,
            'curtailment_rate': 0,
            'h2_production': 0,
            'chp_heat_total': 0,
            'green_power_total': 0,
            'green_power_used': 0,
            'grid_import_total': 0,
            'grid_export_total': 0,
            # 绿氢属性
            'green_power_ratio': 0,
            # 收益项
            'total_revenue': 0,
            'h2_revenue': 0,
            'heat_revenue': 0,
            'grid_export_revenue': 0,
            # 成本项
            'total_cost': 0,
            'grid_import_cost': 0,
            'operation_cost': 0,
            'h2_operation_cost': 0,
            'wind_om_cost': 0,
            'solar_om_cost': 0,
            'depreciation_cost': 0,
            'land_cost': 0,
            'financial_cost': 0,
            'net_economic_benefit': 0,
            # 投资估算
            'total_investment_est': 0,
            # 环保指标
            'carbon_savings': 0,
            # 设备利用率
            'h2_utilization': 0,
            'electrolyzer_utilization': 0,
            'chp_utilization': 0
        }
    def compare_scenarios(self, scenario1_results, scenario2_results, scenario_names=None):
        """对比两个场景的性能"""
        metrics1 = self.evaluate(scenario1_results, [], [])
        metrics2 = self.evaluate(scenario2_results, [], [])
        if scenario_names is None:
            scenario_names = ['Scenario 1', 'Scenario 2']
        comparison = pd.DataFrame({
            'Metric': list(metrics1.keys()),
            scenario_names[0]: list(metrics1.values()),
            scenario_names[1]: list(metrics2.values())
        })
        comparison['Difference'] = comparison[scenario_names[1]] - comparison[scenario_names[0]]
        comparison['% Change'] = (comparison['Difference'] / comparison[scenario_names[0]] * 100).round(2)
        print("\n场景对比分析:")
        print(comparison.to_string(index=False))
        return comparison
    def generate_report(self, metrics, dispatch_results):
        """生成评估报告"""
        report = f"""
# 系统性能评估报告

## 1. 系统场景说明
本系统为**并网型风光制氢热电联产系统**，电网作为备用调节电源：
- 风光出力充足时，优先使用绿电制氢，富余电量上网售电；
- 风光出力不足时，从电网购电补充，保证电解槽连续稳定运行；
- 电解槽运行余热全部回收用于供热，实现能量梯级利用。

## 2. 绿氢属性
| 指标 | 值 | 说明 |
|------|-----|------|
| 绿电占比 | {metrics['green_power_ratio']:.2f}% | 电解槽用电中绿电比例，用于绿氢认证参考 |
| 绿氢产量 | {metrics['h2_production']:.2f} kg | 由绿电生产的氢气总量 |

> **绿氢认证说明**：根据国内绿氢认证相关规范，当电解槽用电中绿电占比≥95%时，可认定为绿氢。本系统绿电占比达 {metrics['green_power_ratio']:.2f}%，{'已达到' if metrics['green_power_ratio'] >= 95 else '接近'}绿氢认证标准。

## 3. 能效指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 系统综合能效 | {metrics['overall_efficiency']:.2f}% | 氢能+热能 / 电解槽消耗总电能 |
| 绿电自用率 | {metrics['self_use_rate']:.2f}% | 制氢消纳绿电占总绿电比例 |
| 绿电上网率 | {metrics['export_rate']:.2f}% | 富余绿电上网售电比例 |
| 弃风弃光率 | {metrics['curtailment_rate']:.2f}% | 无法消纳且无法上网的废弃电量比例 |

## 4. 生产指标
| 指标 | 值 |
|------|-----|
| 绿氢总产量 | {metrics['h2_production']:.2f} kg |
| 热电联产产热 | {metrics['chp_heat_total']:.2f} kWh |
| 电解槽利用率 | {metrics['electrolyzer_utilization']:.2f}% |
| 热电联产利用率 | {metrics['chp_utilization']:.2f}% |

## 5. 电量平衡
| 指标 | 值 | 说明 |
|------|-----|------|
| 总绿电量 | {metrics['green_power_total']:.2f} kWh | 风电+光伏发电总量 |
| 绿电自用 | {metrics['green_power_used']:.2f} kWh | 制氢消耗的绿电量 |
| 电网购电 | {metrics['grid_import_total']:.2f} kWh | 从电网购买的补充电量 |
| 上网售电 | {metrics['grid_export_total']:.2f} kWh | 富余绿电上网售卖量 |

## 6. 经济指标（全成本核算）
### 6.1 收益明细
| 指标 | 值 |
|------|-----|
| 总收益 | {metrics['total_revenue']:.2f} 元 |
| 绿氢收益 | {metrics['h2_revenue']:.2f} 元 |
| 供热收益 | {metrics['heat_revenue']:.2f} 元 |
| 上网售电收益 | {metrics['grid_export_revenue']:.2f} 元 |

### 6.2 成本明细
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

### 6.3 净效益
| 指标 | 值 |
|------|-----|
| 净经济效益 | {metrics['net_economic_benefit']:.2f} 元 |

### 6.4 投资估算
| 指标 | 值 | 说明 |
|------|-----|------|
| 系统总投资估算 | {metrics['total_investment_est'] / 10000:.2f} 万元 | 含电解槽、风电、光伏设备投资 |

> **成本核算说明**：本评估采用全成本核算方法，包含设备折旧、运维成本、土地成本、财务成本等全口径成本。设备折旧按直线法计算，电解槽折旧年限10年，风电20年，光伏25年。风光装机容量按调度周期内最大功率估算，实际投资需根据具体装机规模调整。

## 7. 环保指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 碳减排量 | {metrics['carbon_savings']:.2f} kg CO2 | 绿氢替代灰氢+余热替代燃煤供热 |

> **核算方法**：碳减排量 = 绿氢产量 × 10 kgCO2/kg（灰氢排放因子） + 回收余热 × 0.11 kgCO2/kWh（燃煤供热排放因子）。仅绿电部分产生减排，网电部分不计入。

## 8. 结论
该多能互补系统通过"绿电制氢+热电联产"的协同调度，实现了：
- **{metrics['self_use_rate']:.1f}%** 的绿电自用率
- 系统综合能效达 **{metrics['overall_efficiency']:.1f}%**
- 绿电占比 **{metrics['green_power_ratio']:.1f}%**，{'已达到' if metrics['green_power_ratio'] >= 95 else '接近'}绿氢认证标准
- 累计碳减排 **{metrics['carbon_savings']:.1f} kg CO2**

## 9. 改进建议
1. 扩容电解槽容量，提升绿电消纳比例，降低弃电率
2. 引入预测不确定性处理，优化调度鲁棒性
3. 升级为强化学习调度，自适应优化多目标策略
4. 拓展供热用户，提升余热回收的经济价值
5. 配置储能系统，利用峰谷电价差进行套利，提升经济性
6. 参与绿电交易市场，获取绿电溢价收益
"""
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        print(f"✓ 评估报告已保存至 {self.output_dir}/evaluation_report.md")
        return report
if __name__ == '__main__':
    # 测试评估器
    evaluator = SystemEvaluator(
        electrolyzer_rated=500,
        chp_rated=150,
        h2_storage_rated=1000,
        electrolyzer_investment=1000,
        wind_investment=7000,
        solar_investment=4500,
        electrolyzer_lifetime=10,
        wind_lifetime=20,
        solar_lifetime=25,
        wind_om_cost=0.05,
        solar_om_cost=0.03,
        h2_om_ratio=0.10,
        land_cost_ratio=0.02,
        financial_cost_ratio=0.04
    )
    # 模拟数据
    np.random.seed(42)
    n = 168  # 一周
    dispatch_results = []
    for i in range(n):
        wind_power = np.random.uniform(200, 600)
        solar_power = np.random.uniform(0, 400) if 6 <= i % 24 <= 18 else 0
        total_green = wind_power + solar_power
        electrolyzer_power = 500  # 满负荷运行
        green_for_h2 = min(total_green, electrolyzer_power)
        grid_import = max(0, electrolyzer_power - total_green)
        grid_export = max(0, total_green - electrolyzer_power)
        h2_produced = electrolyzer_power / 50  # 50 kWh/kg
        chp_heat = electrolyzer_power * 0.3 * 0.75  # 废热30%，回收效率75%

        h2_revenue = h2_produced * 30
        heat_revenue = chp_heat * 0.3
        grid_export_revenue = grid_export * 0.35
        grid_import_cost = grid_import * 0.5
        gross_benefit = h2_revenue + heat_revenue + grid_export_revenue - grid_import_cost

        green_power_ratio = green_for_h2 / electrolyzer_power if electrolyzer_power > 0 else 0

        result = {
            'wind_power': wind_power,
            'solar_power': solar_power,
            'electrolyzer_power': electrolyzer_power,
            'green_for_h2': green_for_h2,
            'chp_power': 0,
            'total_green_power': total_green,
            'h2_produced': h2_produced,
            'chp_heat': chp_heat,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'curtailment': 0,
            'h2_storage_level': np.random.uniform(200, 800),
            'h2_revenue': h2_revenue,
            'heat_revenue': heat_revenue,
            'grid_export_revenue': grid_export_revenue,
            'grid_import_cost': grid_import_cost,
            'gross_economic_benefit': gross_benefit,
            'carbon_savings': green_for_h2 / 50 * 10 + green_for_h2 * 0.3 * 0.75 * 0.11,
            'green_power_ratio': green_power_ratio,
            'grid_buy_price': 0.5,
            'grid_sell_price': 0.35
        }
        dispatch_results.append(result)
    metrics = evaluator.evaluate(dispatch_results, [], [])
    print(f"\n测试完成，系统能效: {metrics['overall_efficiency']:.2f}%")
    print(f"绿电占比: {metrics['green_power_ratio']:.2f}%")
    print(f"净经济效益: {metrics['net_economic_benefit']:.2f} 元")
    print(f"总投资估算: {metrics['total_investment_est'] / 10000:.2f} 万元")
