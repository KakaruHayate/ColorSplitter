# ColorSplitter

[English](README.md)

面向**单说话人歌声数据**的音色聚类与筛选工具。

给定一个目录，它会递归找出其中所有音频，逐个提取音色 embedding 并聚类，然后用一张可交互的散点图把结果摆在你面前 —— 你可以点任意一个点直接试听、手动调整簇归属，最后按簇把文件写回磁盘。

典型用途：训练前把一位歌手的数据按音区/风格拆开，或筛掉音色与整体不一致的录音。

**需要说明的一点**：本工具把说话人确认（speaker verification）的技术用在了歌声上。歌声的音色变化与声纹差异相关，但并不等同，这一领域尚无定论。它足够有用，但不是一个已解决的问题。

---

## 安装

```bash
pip install -e ".[all]"          # 全部功能
pip install -e .                 # 仅音色编码器，不装 torch
```

可选依赖组：`cluster`（UMAP + HDBSCAN）、`emotion`、`train`、`web`、`vad`（与参考实现一致的静音裁剪）、`dev`。

读取 `.m4a` / `.mp3` / `.flac` 等格式需要解码器：`PATH` 上有 `ffmpeg` 即可通吃；`libsndfile`（随 `soundfile` 附带）覆盖大部分。

## 使用

```bash
cs serve                          # 浏览器界面，http://127.0.0.1:8000
```

或走命令行：

```bash
cs scan  ./my-singing-data                     # 看看抓到了什么
cs run   ./my-singing-data --nmin 2            # 提特征、聚类，导出 CSV
cs run   ./my-singing-data --nmin 2 --export   # 同时按簇写回磁盘
cs weights list                                # 有哪些权重可用
```

**推荐用 WebUI**：试听和改簇都在那里完成。它做的每件事 `cs` 也都能做，所以工具依然可脚本化。

## 工作流程

```
扫描 → 提特征 → 聚类 → 降维 → 审阅 → 导出
```

* **扫描** —— 递归遍历你指定的目录。**不假设任何数据集结构**：不读标注文件，不要求固定目录名，只有音频本身重要。
* **提特征** —— 每个文件得到一个向量。四种编码器：`timbre`（默认）、`speaker`、`emotion`、`mix`。
* **聚类** —— 谱聚类，或 UMAP + HDBSCAN。算法固定不改，原因见 [docs/design.md](docs/design.md)。
* **降维** —— 给你看的二维投影（t-SNE / UMAP / PCA）。
* **审阅** —— 点选试听、框选批量、改簇归属、改名、合并、拆分、撤销。
* **导出** —— 默认 copy，可选 move，落到 `output/<簇号>/`。

## 权重

权重**不进仓库**：首次使用时下载到本地缓存，并做 SHA-256 校验。清单见 `models/registry.json`。

| id | 用途 | 说明 |
|---|---|---|
| `timbre-v1` | 音色 | 默认，在 `pretrain/` |
| `timbre-alt-v1` | 音色 | 备选 checkpoint，区分度较弱 |
| `speaker-upstream-v1` | 说话人身份 | 上游 Resemblyzer 官方权重，按需下载 |

`timbre` 和 `speaker` 回答的是**不同的问题** —— 前者分离同一歌手的音区，后者区分不同歌手。按需选择。

权重来源、缓存与镜像见 [docs/weights.md](docs/weights.md)。

## 训练

训练代码位于 `src/colorsplitter/training/`，完整可运行；本仓库**不包含任何训练过程**。它期望目录命名为 `<歌手>_<音色>`，其采样器会刻意让每个 batch 里包含**同一位歌手**的多个音色 —— 这些最难区分的样本对，正是教会模型"音色"这一维度的关键。

详见 [docs/training.md](docs/training.md)。

## 文档

| | |
|---|---|
| [installation.md](docs/installation.md) | 环境、可选依赖、解码器、排错 |
| [cli.md](docs/cli.md) | 全部命令与参数 |
| [webui.md](docs/webui.md) | 界面说明与审阅流程 |
| [training.md](docs/training.md) | 数据集布局、采样器、配置、断点续训 |
| [weights.md](docs/weights.md) | 注册表、缓存、镜像、来源 |
| [design.md](docs/design.md) | 架构、保留了什么、重写了什么 |

## 许可

MIT，见 [LICENSE](LICENSE)。第三方出处见 [NOTICE](NOTICE)。
