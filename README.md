# 彩票数据分析与趋势实验室

本项目用于抓取双色球、超级大乐透与福彩3D历史数据，生成 Excel，并在 Streamlit 界面中进行多维可视化分析与数学创意预测展示。



## 功能概览
- 抓取双色球/大乐透/福彩3D历史开奖数据并保存为 Excel
- 支持选择最近 100 期或自定义期数进行分析
- 输出多种走势图（频次、走势、和值、跨度、奇偶结构等）
- 提供多种数学创意预测方法并给出综合推荐号码
- 提供独立的马尔科夫预测分析页，支持双色球 / 大乐透 / 福彩3D
- 图表保存为 JPG（300 DPI），文字为英文，Times New Roman 字体

## 环境要求
- Python 3.10
- Windows / macOS / Linux

## 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始
### 1. 下载/更新历史数据
```bash
python fetch_ssq_history.py
```

自定义期数与输出路径：
```bash
python fetch_ssq_history.py --limit 5000 --output ssq_history.xlsx
```

福彩3D 历史数据：
```bash
python fetch_sd_history.py --limit 5000 --output sd_history.xlsx
```

大乐透历史数据：
```bash
python fetch_dlt_history.py --limit 5000 --output dlt_history.xlsx
```

### 2. 启动 Streamlit 界面
```bash
streamlit run streamlit_app.py
```

进入界面后可在侧边栏：
- 下载/更新历史数据
- 选择分析期数（最近 100 期或自定义）
- 选择彩种（双色球 / 大乐透 / 福彩3D）
- 查看最新一期号码、走势图与预测结果
- 在“马尔科夫预测”页查看专项概率表、推荐号码与导出图表

## 输出说明
- `ssq_history.xlsx`：双色球历史数据
- `dlt_history.xlsx`：大乐透历史数据
- `sd_history.xlsx`：福彩3D历史数据
- `plots/`：自动保存的走势图图片（JPG，300 DPI）

说明：
- 新生成的图像文件名会自动带上数据源文件名和当前时间（精确到秒）
- 马尔科夫分析页面会复用现有历史数据文件，无需额外依赖

## 项目结构
```
.
├── fetch_ssq_history.py     # 双色球数据抓取与 Excel 生成
├── fetch_dlt_history.py     # 大乐透数据抓取与 Excel 生成
├── fetch_sd_history.py      # 福彩3D数据抓取与 Excel 生成
├── streamlit_app.py         # Streamlit 可视化界面
├── ssq_history.xlsx         # 历史数据（可重新生成）
├── dlt_history.xlsx         # 历史数据（可重新生成）
├── sd_history.xlsx          # 历史数据（可重新生成）
├── plots/                   # 图像输出目录（自动创建）
├── requirements.txt         # Python 依赖
└── README.md                # 使用说明
```

## 预测方法说明（简要）
- 分段热平衡（冷热反向权重）
- 间隔波动（偏好中等空档）
- 反聚类协同网络（弱相关组合）
- 马尔可夫转移（状态概率）
- 贝叶斯更新（后验均值）
- 多项/泊松稳定度
- 时间序列趋势（和值预测）
- 互信息网络（弱关联）
- 组合优化（多目标约束）
- Bootstrap/Monte Carlo 模拟
- 波动回归（和值均值回归）
- 复杂系统相空间类比
- 指数记忆热度（近期高权重）
- 周期回归（间隔接近均值）
- 镜像映射（对称扰动）
- 综合投票推荐（多方法投票 + 热度微调 + 分区约束）

## 免责声明
本项目所有预测结果仅供学习与娱乐展示，不构成任何投注建议，请理性对待彩票。

## 常见问题
1. 若出现中文乱码，请确认文件编码为 UTF-8，并使用支持 UTF-8 的编辑器查看。
2. 若图像未保存，请确认当前目录具备写入权限。
