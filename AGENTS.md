# AGENTS.md

## 项目概况

二维分形几何可视化工具，核心算法是迭代函数系统（IFS）与混沌游戏。仓库只有一个核心模块 `IFS.py`，包含：

- `IFSVisualizer`：点集生成与绘制。
- 预置 IFS：蕨、谢尔宾斯基三角形、科赫曲线、龙形曲线、莱维 C 曲线等。
- `RandomFractalGenerator`：随机生成压缩仿射映射。

## 常用命令

```bash
python IFS.py                              # 生成金色蕨头像 IFS.png
python -m unittest discover -s tests -v    # 运行测试
python benchmarks/benchmark.py             # 生成与导出基准
```

## 性能架构

`generate_points` 的 `method` 参数支持四个值：

- `auto`：优先 Numba，仿射系统其次 NumPy，最后纯 Python。
- `numba`：编译后的单条混沌游戏轨道。
- `numpy`：多条独立仿射链，需要函数能被识别为仿射映射。
- `python`：原始逐点循环。

仿射映射在首次调用时通过少量探测点自动识别，识别结果缓存在 `_affine`。非仿射映射（例如 `my_ifs` 中的正弦项）会尝试用 Numba 编译每个函数和一个整数分发器；编译失败时回退到纯 Python。

绘制方面：

- `plot_matplotlib(mode="scatter")` 保持原始散点效果，适合作为最终高质量图。
- `plot_matplotlib(mode="density")` 使用 `histogram2d` 加对数色标，速度和文件体积都优于散点图，适合百万级点集。
- `plot_plotly(max_points=...)` 可以按步长降采样，避免生成过大的 HTML。

基准数据记录在 `README.md` 和 `README.zh-CN.md` 的 Performance 章节。修改性能相关代码后应同步更新这两个章节。

## 开发约定

- 保持 `IFS.py` 位于仓库根目录，README 中的 `python IFS.py` 用法不能失效。
- 新增或修改算法时，在 `tests/` 下补充对应测试。
- 不要提交 `examples/` 中的大体积 PNG（当前约 600 MB）。如需更新示例，优先使用 `assets/preview/` 下的 WebP。
- 提交信息使用英文 Conventional Commits，例如 `perf: compile chaos game with numba`。
