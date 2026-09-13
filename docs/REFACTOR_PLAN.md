# ColorSplitter 重构方案

> **状态：已实施完毕。** 本文是当时用来评审的计划，保留作为决策记录。
> 实施结果与当前架构见 [`design.md`](design.md)，使用方式见 [`../README_CN.md`](../README_CN.md)。
>
> 实施过程中相对本计划的三处偏离，均已在上文或代码注释中说明：
> 1. **不再保留旧脚本薄封装**（原 D6）——用户后续明确要求"小工具全砍掉"，故直接删除。
> 2. **额外移除了 `transformers` 依赖**（用户后续要求），改用自研 wav2vec2 实现，权重按 HF 命名约定加载。
> 3. **训练链路只交付代码并做冒烟验证**，数据集相关统计与路径按要求不写入仓库任何文件。

> 目标：把当前"脚本集合"重写为**单一核心 + CLI + WebUI** 的工程化项目。保留聚类算法与模型架构，重写其余全部。

---

## 一、现状汇总

### 1.1 仓库实测快照

| 文件 | 行数 | 职责 | 处置 |
|---|---|---|---|
| `splitter.py` | 60 | 提特征 + 聚类 + 出图，`input()` 交互循环 | **删除**，能力迁入 CLI / WebUI |
| `move_files.py` | 20 | 按 CSV 把音频 **copy** 到簇目录 | **删除** |
| `kick.py` | 21 | 把指定簇音频 **move** 出原数据集 | **删除** |
| `clean_csv.py` | 11 | 原地删标注文件里无对应音频的行 | **删除** |
| `load_npy.py` | 37 | 读 `.npy` 画图（不聚类） | **删除** |
| `modules/utils.py` | 79 | 三种编码器 + pkl 缓存 | **重写** |
| `modules/cluster.py` | 169 | SpectralCluster（3D-Speaker 移植）+ UmapHdbscan | **保留**，仅等价加速 |
| `modules/visualizations.py` | 217 | 投影可视化（Resemblyzer 移植） | **删除**，可视化由前端承担 |
| `modules/model/voice_encoder.py` | 153 | Resemblyzer 系 VoiceEncoder | **保留定义**，去依赖 |
| `modules/model/emotion_encoder.py` | 70 | wav2vec2 情绪编码器 | **保留**，改按需加载 |
| `viewer/` | ~600 | React + Vite 散点图 + 试听 | **重写**为随主程序分发的 WebUI |
| 合计 | ~800 行 Python + ~600 行 TS | | 无测试 / 无 CI / 无打包 |

### 1.2 保留的代码资产（重写的锚点）

只有三样东西值得留，其余按新架构重写：

| 资产 | 为什么留 |
|---|---|
| `cluster.py` 的 `SpectralCluster` / `UmapHdbscan` | **算法是特挑的，不改**。整个项目最核心的资产 |
| `VoiceEncoder` 的模型定义 | 权重兼容性锚点：所有历史权重都依此结构（3 层单向 LSTM / hidden 256 / mel 40 / `linear(256→256)`） |
| `emotion_encoder` 的模型定义 | 情绪编码能力的定义 |

### 1.3 权重盘点（实测）

| | 权重 A（社区惯用） | 权重 B（合规备选） |
|---|---|---|
| 位置 | 被覆盖在 git 历史中，可无损取回 | 当前工作区版本 |
| blob | `e8560833efe9e9e0f0bda3cba03311a0fe5f201f` | `f25b814ed4b69a1c7b61f6f956947100abd51a0f` |
| 大小 | 17,100,415 B | 17,095,286 B |
| 内部 `step` | **1,570,001** | **165,000** |
| `model_state` | 16 个张量 | 16 个张量（形状完全一致） |
| 架构 | 3 层单向 LSTM / hidden 256 / mel 40 / `linear(256,256)` | 完全相同 |
| 附加 | `optimizer_state` + `similarity_weight/bias` | 同 |

结论：

1. **权重 A 没有被删除**，只是被一次信息为 "Update README.md" 的提交连同 README 一起覆盖掉了，所以从提交列表上看不出来。用上面的 blob 哈希即可无损取回。
2. 两个权重**逐层同构**，可被同一份模型定义加载，"双权重并存、默认 A"在工程上零障碍。
3. 当前文件名（`encoder_1570000.bak`）与权重 B 的实际训练步数（165,000）**不符**，会误导后续判断，须重命名。
4. 两者都携带完整 `optimizer_state`（zip member 57 / 39），作为推理权重分发属冗余，应剥离为纯 `model_state`。

另需引入第三个权重：上游 Resemblyzer 官方 `pretrained.pt`，作为**通用说话人分类**模型（权重 A/B 优化的是"音色风格区分"，官方权重优化的是"说话人身份区分"，两者用途不同，并存不冲突）。

### 1.4 训练代码

仓库内不存在（历史树中从未出现），确认按**重写**处理。原始血脉为 `Real-Time-Voice-Cloning` 的 speaker encoder 训练（GE2E 目标），当初改动点在 dataloader：**每个 batch 尽量取同一歌手的多个不同音色**，即硬负样本采样。

### 1.5 数据与合规约束（硬性）

训练数据为外部单说话人歌声数据集，其许可状况存在争议。

> **约束**：该数据集的来源、标识、路径、统计数字**一律不写入仓库内任何文件** —— 包括文档、注释、示例、测试夹具、CI 配置、错误信息。
> 代码只依赖**目录命名约定**（`<歌手>_<音色>/…`），不依赖任何具体数据集。

### 1.6 缺陷清单

绝大多数缺陷位于**即将被删除的脚本**中，因此不需要"修"，只需要"不带走"：

| 位置 | 问题 | 处置 |
|---|---|---|
| `splitter.py` | `--mer_cosine` 完全失效（取值后赋给局部变量，传给 `CommonClustering` 的永远是 `mer_cos=None`） | 随脚本删除；新实现中该参数必须接通 |
| `splitter.py` | `--mer_cosine` 声明为 `type=str`，与 `assert cos_thr > 0` 冲突，接通即 `TypeError` | 同上，新实现用 `float` |
| `splitter.py` | 只匹配 `*.wav`，与常见音频格式不匹配 | 新实现递归扫描多格式 |
| `splitter.py` | 靠 `input()` + `plt.show()` 阻塞，无法批处理、无法被程序调用 | 新实现为纯函数编排层 |
| `move_files.py` / `kick.py` | copy 与 move 语义互相矛盾，且 move 会把文件搬出原数据集（破坏性副作用） | 收敛为显式 `copy` / `move` 两种导出模式 |
| `clean_csv.py` | 原地改写标注文件且无备份 | 移除；改为只读校验 + 清单导出 |
| `modules/utils.py` | 特征缓存 `*.pkl` 混在数据目录内，无失效机制 | 重写为独立缓存目录 + 内容哈希键 |
| `modules/model/emotion_encoder.py` | `import` 期即加载模型（未使用情绪功能也会强制加载/下载） | 改按需加载 |
| `viewer/` | 需并行运行第二个静态服务器 + CORS，且前端用绝对路径拼音频 URL | 重写为单端口同源服务 |
| 全局 | `requirements.txt` 缺 `scipy`、`librosa`（代码直接 import） | 重写依赖清单并锁版本 |
| `cluster.py` | `max_num_spks=14` 写死簇数上限，与 `--nmin` 会冲突 | 保留算法，但把上限改为可配置 |
| 全局 | 无测试、无 CI、无打包、无类型标注、无日志 | 随骨架一并建立 |

### 1.7 性能瓶颈

**推理（embedding 提取）**
- 现状是逐文件串行 `preprocess_wav` → `embed_utterance`；每个 utterance 又被切成约 `1.3 × 时长` 个 1.6 s partial，逐 partial、batch=1 过 LSTM。
- 三重浪费：① batch=1 前向，GPU 基本空转；② wav→mel 走 numpy 单线程；③ 压缩音频解码串行，IO 是硬瓶颈。
- 优化方向：partial 级批量前向、多进程并发解码、mel 与 embedding 缓存、GPU + 半精度路径。

**聚类**
- `SpectralCluster` 对 **N×N 稠密**亲和矩阵做 `scipy.linalg.eigh` → 全量特征分解 O(N³)；`p_pruning` 再对每行做一次全量 `argsort`。
- `UmapHdbscan` 默认 `n_components=60`，UMAP 在 60 维上极慢。
- 优化方向（**算法不变、结果等价，以对拍验证**）：① kNN 稀疏图 + `scipy.sparse.linalg.eigsh`(ARPACK) 只求前 k 个特征向量；② UMAP 前先 PCA 降到 ~50 维；③ 相似度矩阵分块/向量化替代逐行循环。

**降维（可视化）**：2D 投影每次重算，应缓存复用。

---

## 二、目标与原则

| 原则 | 说明 |
|---|---|
| **单一核心** | CLI 与 WebUI 共用同一套 `core`，不存在"只有某个入口能做的操作" |
| **工具全部收敛到 WebUI** | 原 `move_files` / `kick` / `clean_csv` / `load_npy` 的能力全部并入 WebUI 与 CLI，不再保留独立脚本 |
| **输入即目录** | 输入 = 一个目录，递归扫描 `wav/mp3/flac/m4a/ogg`。不引入任何数据集结构假设 |
| **去 Resemblyzer 依赖** | 不再依赖 `Resemblyzer` 包；把它实际用到的部分（重采样预处理、mel 频谱、hparams 常量）内联进 `core/audio.py`，并按 MIT 保留出处声明 |
| **算法守恒** | 聚类与降维算法不换，只做等价加速，用对拍测试证明等价 |
| **训练只交付代码** | 训练链路完整可用、可 smoke test，但**不进行实际训练**，不产出新权重 |
| **双权重 + 第三权重** | 权重 A 为默认；权重 B 为合规备选；另加官方说话人模型用于说话人身份区分 |
| **不泄露** | 不写入会话背景、不写入本地绝对路径、不写入数据集标识与统计 |
| **可观测** | 长任务必须有进度上报（CLI 进度条 / Web SSE） |

---

## 三、目标架构

```
ColorSplitter/
├─ pyproject.toml                 # 打包 + 锁版本依赖
├─ NOTICE                         # 第三方出处与许可（3D-Speaker / Resemblyzer / wav2vec2）
├─ src/colorsplitter/
│  ├─ core/
│  │  ├─ audio.py                 # 递归扫描 + 解码 + 重采样 + mel（内联自 Resemblyzer，MIT）
│  │  ├─ embed.py                 # timbre / speaker / emotion / mix；partial 批量化 + 缓存
│  │  ├─ reduce.py                # t-SNE / UMAP 2D 投影（可缓存）
│  │  ├─ cluster.py               # SpectralCluster / UmapHdbscan（等价加速版）
│  │  ├─ labels.py                # 簇编辑：改点归属、合并、拆分、改名（纯函数，可单测）
│  │  ├─ pipeline.py              # scan → embed → cluster → project → export 编排（无 input()）
│  │  ├─ modelzoo.py              # 权重注册表 + 自动下载 + SHA256 校验
│  │  └─ types.py
│  ├─ cli/                        # cs scan / embed / cluster / export / serve / train
│  ├─ web/
│  │  ├─ app.py                   # FastAPI：/api/*、SSE 进度、/media 音频流（单端口）
│  │  └─ static/                  # 前端构建产物
│  └─ training/                   # 训练链路（交付代码，不做实际训练）
│     ├─ data.py                  # <歌手>_<音色> 数据集 + GE2E 采样器
│     ├─ model.py                 # 与推理端同构的 VoiceEncoder
│     ├─ loss.py                  # GE2E
│     ├─ train.py                 # 训练循环 + 断点续训
│     └─ configs/*.yaml
├─ frontend/                      # 前端源码（构建产物进 web/static）
├─ models/
│  ├─ registry.json               # 权重注册表（id / 步数 / 用途 / 许可说明 / SHA256 / 下载源）
│  └─ .cache/                     # 运行时下载（gitignore，不入库）
├─ tests/                         # 含"等价加速"对拍测试
└─ docs/
```

**分层**

- **入口层**：CLI（脚本化 / CI）与 WebUI（交互）能力对等。
- **编排层**：`pipeline.py` 只做编排，全部可单测；无全局状态、无阻塞交互。
- **核心层**：音频、编码、聚类、降维、簇编辑五个纯能力模块。
- **横切**：模型库、特征缓存、配置。
- **产出层**：训练、导出、Web 服务。

---

## 四、WebUI 功能规格

### 4.1 核心交互（本次明确要求）

| 功能 | 行为 |
|---|---|
| **点选试听** | 点击散点图中任一数据点，立即播放该点对应的音频。播放器常驻（全局唯一实例），重复点击同一点为播放/暂停切换。音频经同源 `/media` 流式播放，**不暴露磁盘路径**（用短 ID 映射） |
| **编辑点的簇归属** | 选中一个或多个点后，可直接改写其簇序号；支持：① 单点改簇；② 框选/多选批量改簇；③ 拖拽出一个新簇；④ 撤销/重做 |
| 簇级操作 | 改名、合并两簇、把某簇拆出子集、删除空簇 |
| 实时联动 | 任何簇编辑立即反映到散点图配色、右侧列表与统计；导出前始终以当前编辑状态为准 |

### 4.2 原脚本能力的落点

| 原脚本 | WebUI / CLI 落点 |
|---|---|
| `move_files.py` | 「导出」：按簇落盘，**copy** 模式 |
| `kick.py` | 「导出」：**move** 模式（需二次确认，明确提示会移动源文件） |
| `clean_csv.py` | 「校验」：只读检查音频清单完整性并导出报告，不再改写任何原文件 |
| `load_npy.py` | 「导入 embedding」：直接载入已有 `.npy` 做降维/聚类/可视化，跳过推理 |
| `splitter.py` | 主流程：扫描 → 提取 → 聚类 → 可视化 → 试听 → 编辑 → 导出 |

### 4.3 其余功能

- 散点图支持 t-SNE / UMAP 切换，2D 投影缓存复用。
- 长任务（提取、聚类）走 SSE 上报进度，可取消。
- 导出 CSV（含当前簇标签与 2D 坐标）。
- 训练**不纳入** WebUI。

---

## 五、分阶段计划

> 顺序即依赖。每阶段有验收标准，不通过不进入下一阶段。

### 阶段 0 — 骨架与清理

- 建立 `pyproject.toml`、`src/` 布局、依赖清单（**移除 `Resemblyzer`**）、CI（lint + test）、`tests/` 骨架。
- 删除 `splitter.py` / `move_files.py` / `kick.py` / `clean_csv.py` / `load_npy.py` / `modules/visualizations.py` / `viewer/`。
- 把 `cluster.py`、`VoiceEncoder` 定义、`emotion_encoder` 定义迁移进 `core/` 与 `training/`，去掉对 `resemblyzer` 包的 import，内联所需部分到 `core/audio.py`。
- **验收**：CI 绿；`import colorsplitter` 在干净环境成功且**未安装** `Resemblyzer`。

### 阶段 1 — core 抽取与解耦

- `core/` 五模块成型，全部纯函数化；输入模型为"目录 → 递归扫描音频"。
- 缓存移出数据目录，改为内容哈希键 + 独立缓存目录。
- `modelzoo.py` 打通：从 git 历史恢复权重 A，剥离 `optimizer_state`，按真实步数重命名；权重 A/B 与官方说话人模型三者入注册表。
- **验收**：对任一含音频的目录，`cs scan` / `cs embed` / `cs cluster` 全链路可跑通；同一输入下簇划分与 `cluster.py` 原始实现一致。

### 阶段 2 — 性能

- 推理：partial 级批量化前向、多进程并发解码、mel 与 embedding 缓存、GPU/半精度路径。
- 聚类：kNN 稀疏图 + ARPACK；`p_pruning` 向量化。
- 降维：UMAP 前 PCA 预降维；2D 投影缓存。
- 建立基准脚本，固定输入、分段记录 scan / embed / cluster / project 耗时。
- **验收**：对拍测试证明簇标签与投影与旧实现一致（差异超阈值即回退）；给出改动前后的分段耗时对照。

### 阶段 3 — 权重与模型库

- `modelzoo.py`：`registry.json` 驱动的解析 + 自动下载（官方源 + 镜像回退）+ SHA256 校验 + 断点续传。
- 三个权重可用：A（默认）、B（备选）、官方说话人模型；emotion 模型按需加载并自动下载。
- `docs/` 与 `registry.json` 写明权重 A 的许可说明：其训练数据用于 **embedding 模型**、未进入任何合成路径。
- **验收**：干净 clone 后仅凭 `registry.json` 可自动获取并校验全部权重；仓库内**不含权重二进制**。

### 阶段 4 — 训练链路（只交付代码）

- `training/data.py`：扫描 `<歌手>_<音色>` 目录；GE2E 采样器按"每 batch 优先取同一歌手的多个不同音色"构造硬负样本。
- `training/model.py` 与推理端同构（保证权重可互换）。
- `training/loss.py`（GE2E）+ `training/train.py`（配置化、断点续训、checkpoint 只存 `model_state`）。
- **本阶段不进行实际训练**，不产出新权重。
- **验收**：用合成的小规模临时数据可完整跑通若干 step（smoke test）；单测覆盖采样器的批次构成约束（同歌手异音色占比）；产出的 checkpoint 可被 `core/embed.py` 加载。

### 阶段 5 — WebUI

- `web/app.py`：FastAPI 单端口同时提供 API、音频流与前端静态资源。
- 前端实现 4.1 全部交互 + 4.2 全部落点。
- **验收**：单条命令启动，浏览器完成"扫描 → 提取 → 聚类 → 点选试听 → 改簇归属 → 导出"全流程，无需任何命令行操作。

### 阶段 6 — 文档与发布

- 重写 `README.md` / `README_CN.md`；`docs/` 补：安装、CLI 参考、WebUI 使用、训练指南、算法与出处、许可说明。
- 发布 release，附带三个权重与 emotion 模型。
- **验收**：全新环境按文档从零跑通全流程；文档中无任何本地路径与数据集信息。

---

## 六、接口契约

**`core/pipeline.py`**

```
scan(root, exts) -> AudioDataset
embed(dataset, encoder, weights, device, batch, workers, cache) -> EmbeddingSet
cluster(embeds, method, nmin, mer_cos: float | None, max_num_spks, **kw) -> ClusterResult
project(embeds, method) -> ndarray[N, 2]
relabel(result, changes) -> ClusterResult      # 点/批量改簇
export(dataset, result, dest, mode: "copy" | "move") -> ExportReport
```

`encoder` 取值：`timbre`（默认） / `speaker`（官方说话人模型） / `emotion` / `mix`。

**权重注册表 `models/registry.json`（结构）**

```
{
  "encoders": [
    {"id": "timbre-v1", "file": "…", "step": 1570001, "purpose": "timbre",
     "default": true, "sha256": "…", "note": "训练数据仅用于 embedding，未进入合成路径"},
    {"id": "timbre-alt", "file": "…", "step": 165000, "purpose": "timbre", "default": false},
    {"id": "speaker-official", "file": "…", "purpose": "speaker", "upstream": "Resemblyzer (MIT)"}
  ],
  "downloading": { "emotion": {"repo": "…", "file": "pytorch_model.bin"} }
}
```

**Web API**

```
GET  /api/datasets                     扫描结果
POST /api/embed                        创建任务 → task_id
GET  /api/tasks/{id}/events            SSE 进度
POST /api/cluster                      聚类（method / nmin / mer_cos / max_num_spks）
GET  /api/projection                   2D 投影 + 簇标签
GET  /media/{token}                    音频流（token → 文件映射，不暴露路径）
POST /api/labels                       改点归属（含批量、撤销/重做）
POST /api/clusters/merge|rename|split
POST /api/export                       导出（copy / move）
POST /api/import-embeddings            载入外部 .npy
```

**训练数据集契约**

```
<root>/<歌手>_<音色>/<任意文件名>.{wav,m4a,flac,mp3}
```
类别 = `<歌手>_<音色>`；同一 `<歌手>` 下不同 `<音色>` 互为硬负样本。

---

## 七、设计决策

| # | 决策 | 结论 |
|---|---|---|
| D1 | 训练类别粒度 | `<歌手>_<音色>`。同歌手不同音色互为硬负样本，模型学的就是"区分音色" |
| D2 | 加速边界 | 允许**等价加速**：只替换实现，算法与参数默认值不变，用对拍测试锁死一致性 |
| D3 | WebUI 技术栈 | **纯 Python 单端口**（FastAPI 同端口提供 API + 音频流 + 静态资源），终端用户无需 node |
| D4 | 权重分发 | **release 附件 + 自动下载 + SHA256 校验**；权重不再进 git |
| D5 | 输入契约 | 一个目录，递归扫描多格式音频；不引入数据集结构假设 |
| D6 | 旧脚本处置 | **直接删除**，不保留薄封装；能力全部并入 WebUI / CLI |
| D7 | Resemblyzer 依赖 | **移除包依赖**；内联实际用到的部分（MIT，保留出处）；官方权重作为"说话人分类"模型引入 |
| D8 | 训练范围 | 只交付**可用代码**，不进行实际训练、不产出新权重 |
| D9 | 默认权重 | 权重 A（社区惯用）为默认；权重 B 为合规备选；两者并存可切换 |
| D10 | 信息约束 | 文档、注释、示例、测试夹具中一律不出现会话背景、本地绝对路径、数据集标识与统计 |

---

## 八、剩余风险

| # | 风险 | 处置 |
|---|---|---|
| R1 | 权重 A 的许可属"擦边"（训练数据 EULA 要求不得用于合成路径，而本项目仅作 embedding 无监督分类） | 在 `registry.json` 与文档中如实说明用途边界；同时提供合规备选权重 B，用户可自行选择 |
| R2 | 压缩音频解码依赖外部解码器 | 启动自检 + 明确可执行的报错信息 |
| R3 | 等价加速引入数值差异 | 对拍测试作为阶段 2 准入门槛，超阈值回退 |
| R4 | `pval` / `min_pnum` / `n_components=60` / `min_cluster_size=4` 等默认值可能欠优 | **本次不改**，另立实验项，避免重构与调参互相污染归因 |
| R5 | 权重历史已存在于 git 提交历史中 | 本次不重写历史；后续若需瘦身，另立方案单独评估 |
| R6 | 前端构建产物入库会带来噪音提交 | `web/static/` 仅在发版时更新，并在贡献指南中写明 |

---

## 九、全局验收标准

1. `cs serve` 单端口提供 WebUI；`cs` 子命令覆盖全流程；均无需 `cd` 到特定目录、无需 node。
2. 输入仅要求"一个含音频的目录"。
3. 聚类与降维结果与旧实现**一致**（对拍测试通过）。
4. 三个权重可用、默认为权重 A；权重经 release + 自动下载获取并校验，仓库内不含权重二进制。
5. 训练链路 smoke test 通过、权重可被推理端加载；**不要求**实际训练。
6. CI 绿：lint + 单测 + 对拍测试。
7. 干净环境在**未安装 `Resemblyzer`** 时可完整运行。
8. 全仓库（含文档、注释、示例、测试夹具）不含会话背景、本地绝对路径、数据集标识与统计。
9. 文档从零环境可跑通，含许可与第三方出处声明。
