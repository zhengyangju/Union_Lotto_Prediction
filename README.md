# 双色球数据分析与趋势实验室

本项目用于抓取双色球历史数据，生成 Excel，并在 Streamlit 界面中进行多维可视化分析与数学创意预测展示。

## 功能概览
- 抓取双色球历史开奖数据并保存为 Excel
- 支持选择最近 100 期或自定义期数进行分析
- 输出多种走势图（频次、走势、和值、跨度、奇偶结构等）
- 提供多种数学创意预测方法，并给出综合推荐号码
- 自动保存图像为 JPG（300 DPI，Times New Roman）

## 环境要求
- Python 3.10
- Windows / macOS / Linux 均可运行

## 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 下载/更新历史数据
```bash
python fetch_ssq_history.py
```

自定义抓取期数与输出路径：
```bash
python fetch_ssq_history.py --limit 5000 --output ssq_history.xlsx
```

### 2. 启动 Streamlit 界面
```bash
streamlit run streamlit_app.py
```

进入界面后可在侧边栏：
- 下载/更新历史数据
- 选择分析期数（最近 100 期或自定义）
- 查看最新一期号码、走势图与预测结果

## 输出说明
- `ssq_history.xlsx`：抓取后的历史数据文件
- `plots/`：自动保存的走势图图片（JPG，300 DPI）

## 项目结构
```
.
├── fetch_ssq_history.py     # 数据抓取与 Excel 生成
├── streamlit_app.py         # Streamlit 可视化界面
├── ssq_history.xlsx         # 历史数据（可重新生成）
├── plots/                   # 输出图像目录（自动创建）
├── requirements.txt         # Python 依赖
└── README.md                # 使用说明
```

## 预测方法说明（简要）
- 分段熵平衡（冷热反向权重）
- 间隔波动（偏好中等空档）
- 反聚类协同网络（弱相关组合）
- 指数记忆热度（近期高权重）
- 周期回归（间隔接近均值）
- 镜像映射（对称扰动）
- 综合投票推荐（多方法投票 + 热度微调 + 分区约束）

## 免责声明
本项目所有预测结果仅供学习与娱乐展示，不构成任何投注建议。请理性对待彩票。

## 常见问题
1. 若出现中文乱码，请确认文件编码为 UTF-8，并使用支持 UTF-8 的编辑器查看。
2. 若图像未保存，请确认当前目录具备写入权限。
