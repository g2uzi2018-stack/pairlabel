# PairLabel · 协标

**使用两个异构本地视觉模型独立识别、交叉复核，把图片转成可检查、可追溯的 YOLO / COCO 数据集。**

类别与补充提示词按任务配置，中国象棋是可选增强预设。提供中文 Web 界面、检测框、SAM 分割轮廓、疑难筛选、自动重试和中断续跑。图片与推理均留在本地。

## 功能

- **通用类别与提示词**：每任务 1–200 类、最多 2000 字补充要求；可预览实际提示词，配置在任务创建时保存快照。
- **独立双模型复核**：A 生成标注，B 先独立识别，再按一对一匹配检查目标、类别和位置；B 不接收 A 的预测。
- **检测与分割**：VLM 生成框，SAM 2.1 Hiera Large 提取轮廓；保留原始框、掩膜自评及不确定原因。
- **批量任务**：上传图片或递归读取本机目录、按内容去重、查看进度、暂停/继续、筛选和单独重试疑难图片。
- **可追溯导出**：YOLO 检测、YOLO 分割、COCO，附预览、独立输出、任务配置和人工复核记录。默认不导出疑难。
- **象棋增强可选**：14 类红黑棋子预设、读字/颜色、圆形候选、数量辅助检查；通用模式不套用这些规则。

本项目负责图片数据集标注。实际应用中的姿态推算由下游程序完成。

## 已验证环境与模型

| 项目 | 已验证配置 |
| --- | --- |
| 设备 | NVIDIA Jetson AGX Orin，64GB 统一内存 |
| 系统 | Ubuntu 22.04 aarch64，L4T 36.4.4，CUDA 12.6，Python 3.10 |
| 生成模型 | Qwen3.6-27B Q6_K + FP16 视觉编码器 |
| 独立复核模型 | InternVL3.5-14B Q8_0 + FP16 视觉编码器 |
| VLM 运行时 | 本地 CUDA llama.cpp，两个模型顺序加载 |
| 分割 | SAM 2.1 Hiera Large，本地 CPU FP32 |

效果优先，没有为了速度默认换成小模型。SAM 使用 CPU 是当前验证环境的选择；切勿把系统已有 PyTorch 擅自替换为不兼容的 CUDA 版本。

当前后端使用 Linux 的 `fcntl`、`/proc` 和本地 llama-server。已验证的是上述 Jetson 配置；其他 Linux/NVIDIA 设备需要自行验证，Windows 可作为浏览器客户端。

## 安装与启动

仓库仅包含源码、测试和许可证，**不包含模型权重、CUDA llama.cpp、用户图片及任务数据库**。安装脚本会直连下载依赖；模型下载需要显式执行，不会在标注时隐式触发。

### 1. 获取源码与 Python 依赖

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/g2uzi2018-stack/pairlabel.git
cd pairlabel
bash scripts/install.sh
```

脚本创建 `.venv`，使用 `--system-site-packages` 以复用已验证的系统环境，不覆盖系统 Python。需要 Python 3.10 的 `venv` 支持；Ubuntu 缺失时先安装对应的 `python3-venv` 包。

### 2. 准备本地模型和 CUDA llama.cpp

本项目不负责自动构建 Jetson 的 CUDA 推理运行时。请先准备支持这两种视觉架构及相应视觉编码器的 llama-server；已验证源码版本见 `MODEL_MANIFEST.json`。

- 默认运行时路径：`~/ai/src/llama.cpp/build/bin/llama-server`。
- 默认 Qwen 权重路径：`~/ai/models/qwen3.6-27b/Qwen3.6-27B-Q6_K.gguf` 与同目录 `mmproj-F16.gguf`。该模型在验证设备上原已存在，下载脚本不会下载或替换它。
- `python3 scripts/download_models.py` 显式下载 InternVL 权重和编码器到本项目 `models/`，支持分块续传并验证 SHA256。
- `bash scripts/install_sam.sh` 安装 CPU SAM 依赖及仓库内 SAM 源码，并直连下载 SAM 2.1 Large 权重。此步骤下载体积较大，请按需执行。

默认镜像、模型来源、版本及已有校验值见 `MODEL_MANIFEST.json`。下载脚本禁用显式代理；不会通过代理下载模型。更换模型时必须同时确认 llama.cpp 的架构与编码器兼容性。

### 3. 启动 Web 界面并核对模型路径

```bash
bash scripts/start.sh
```

本机访问 **http://127.0.0.1:7860**。点击“类别、提示词与模型设置”，在高级 JSON 中按实际位置修改 `runtime.binary`、两模型的 `weights`/`projector` 和 `sam2.weights`。路径使用本机绝对路径。未就绪的模型会在界面显示，准备完成后再处理任务。

也可在启动前生成默认配置再编辑：

```bash
python3 -c 'from app.config import load_config, save_config; save_config(load_config())'
# 按本机实际路径编辑 config/config.json
```

两模型必须使用不同权重和不同视觉架构。默认服务仅监听回环地址，不需要开放公网 Web 端口。通过 SSH 从另一台电脑访问：

```bash
ssh -N -T -L 17860:127.0.0.1:7860 your-user@your-jetson
```

保持通道打开，浏览器访问 http://localhost:17860 。有自定义 SSH 端口或密钥时自行附加 `-p` / `-i`。

### 可选：用户级 systemd 服务

示例假定工程在 `~/workspace/pairlabel`。其他位置先修改 unit 内路径。

```bash
mkdir -p ~/.config/systemd/user logs
cp scripts/pairlabel.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pairlabel
```

需要退出登录后继续运行或开机运行时，再由设备管理员配置用户 linger。

## 使用

1. 新建任务，选择“通用物件”或载入“象棋增强”预设。
2. 每行填写一类，例如 `cat | 猫`、`remote_control | 遥控器`，也可直接写 `杯子`。ID 自动从 0 编号。名称不可重复，不能含竖线、换行或首尾空白；竖线用于分隔导出名称与显示名称。
3. 填写补充要求，例如“仅标注桌面上的目标；框包含完整物体及把手，不包含投影”，点击更新提示词预览。
4. 上传图片或输入 **推理设备上的目录**。勾选是否提取轮廓、是否自动重试，再开始处理。
5. 对照最终标注、模型 A、模型 B，筛选疑难；需要时编辑框/轮廓、人工确认或排除，并填写复核依据。
6. 暂停或完成后导出。可直接下载 ZIP，也可仅保存在设备上，减少大批量远程传输。

支持 JPG、PNG、WebP、BMP、TIFF 等静态图片。单图最多 80MB / 3600 万像素，单任务最多 5000 张。多帧图片需先拆分。EXIF 方向统一后保存为 RGB PNG；不会改写输入目录中的原图。

默认可读取 `~/workspace`、`~/Pictures`、`~/Downloads`。其他输入目录需在设置中添加到 `allowed_roots`；会检查真实路径并跳过逃逸目录的软链接。应用自己的 `data/` 不能作为输入。

## 复核与恢复机制

两个模型按批次顺序加载，接收原始图片、同一类别表和同一标注规范，互不可见对方结果。比较时采用全局一对一 IoU 匹配，保留错类、漏标、位置、未知区域、自报置信值及轮廓质量问题。

通用模式重试使用有重叠的 2×2 原图分块，继承自定义类别及补充要求。象棋模式还支持不含类别提示的像素候选拼图；共享候选的相同框不被当作独立定位证据。无法读字的背面棋子会保留疑难，避免反复猜测。

SQLite WAL 是权威状态；每次独立识别完成后立即保存。中断恢复时复用已保存的成功预测，只补做缺失步骤，保留全部尝试。暂停会在当前模型对当前图的整组推理保存后生效。已有任务保留创建时配置快照，之后改设置不会重新解释旧任务。

SAM 检查掩膜自评、阈值稳定性、面积与定位，原 VLM 框保留为 `vlm_bbox`。GrabCut 是可显式选择的备用后端，`segmentation_backend: "grabcut"`；不会在 SAM 不可用时悄悄替换。

空图默认不会自动通过，需要负样本时明确设置 `allow_empty: true`。

## 导出结构

```text
manifest.json                  类别、配置、文件哈希、纳入/排除清单
README.txt
yolo/dataset.yaml
yolo/images/{train,val}/*.png
yolo/labels/{train,val}/*.txt    class_id cx cy w h，归一化
yolo-seg/...                    class_id + 归一化轮廓点
coco/images/{train,val}/*.png
coco/annotations/instances_{train,val}.json
previews/*.jpg
review/*.json                  每次模型输出与复核证据
review/audit.jsonl              人工操作及任务审计
```

YOLO ID 从 0 开始，COCO 类别 ID 从 1 开始。图片按内容哈希稳定划分训练/验证集，同图不会跨集合；少量样本可能使某个集合为空。YOLO 的图片路径相对于 `dataset.yaml`，解压后可移动。

YOLO 分割中，只要一张图片有目标缺少有效轮廓，就整体跳过该图，避免遗漏目标变成负样本；清单写入 manifest。COCO 中 `segmentation=[]` 表示该目标只有检测框。

默认只导出自动通过或人工确认的结果。显式包含疑难时，请自行承担数据质量复核；导出保留疑难标记与原始分歧。

## 效果与速度：已测结果和限制

**两个模型一致不能证明标注正确。自报置信值、SAM 掩膜自评分及稳定性都不是经过真值校准的准确率。** 类别可任意配置，也不代表模型能可靠识别任意物件；实际数据需要抽样复核。

历史小规模验证：六棋子程序生成图 6/6 类别和框匹配、掩膜平均 IoU 0.920；4 张公开实拍/游戏图均保留疑难。实拍自动通过率偏低，当前不能承诺免人工。

另用一张公开 COCO 猫与遥控器图片跑通通用模式：两模型均检出 2 只猫、2 个遥控器，4 个目标生成轮廓；一个遥控器的两路框 IoU 0.3198，整图保留疑难，未降阈值或人工放行。约耗时：

| 步骤 | 单张验证图耗时 |
| --- | --- |
| Qwen 独立推理 | 57.654 秒 |
| InternVL 独立推理 | 27.122 秒 |
| SAM 图像处理 | 14.792 秒 |
| 任务总时间，含模型顺序加载等开销 | 213.353 秒 |

这些是已测硬件和单张样本的观测，不能外推为吞吐量或泛化准确率。含疑难结果的 YOLO 检测/分割及 COCO 导出已逐项检查类别与坐标。公开评估图、私人数据及任务记录不随源码分发。

## 开发检查

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

发布整理前，Jetson 上 48 项回归通过；涵盖任务快照、双模型真实请求组装、通用/象棋模式隔离、错误输出、疑难处理、断点恢复、分割失败和导出坐标。另完成真实模型及浏览器流程验证。测试不需要下载 VLM 权重；测试中的模拟响应不能作为模型效果证据。

可选的真实模型基准会处理带真值的程序生成象棋图，不能与 Web 推理同时运行：

```bash
.venv/bin/python scripts/make_samples.py
.venv/bin/python scripts/benchmark.py --role generator
.venv/bin/python scripts/benchmark.py --role reviewer
```

生成样图需要 Linux 上的 Noto CJK 字体。`requirements.lock.txt` 只是历史设备环境快照，不是可移植安装锁；安装使用 `requirements.txt` 和对应脚本。

## 数据与备份

- `config/config.json`：本机设置，不提交 Git。
- `data/labeler.sqlite3`：任务、图片和审计的权威状态。
- `data/jobs/<id>/images/`：规范化原图；`results/`：预览及证据。
- `data/exports/`：导出 ZIP；`logs/`：模型与处理日志。

备份时暂停任务并停止服务，再备份 `data/` 和 `config/`；或使用 SQLite backup API。不能只复制活动数据库主文件而漏掉 WAL。目录改名或从旧的 `xiangqi-labeler` 部署迁移时，要先保留数据并核对旧任务快照中的绝对路径，勿直接覆盖现有部署。

## 许可证与组件

应用代码采用 [MIT License](LICENSE)。内置 SAM 2 源码及其他组件保留各自许可证，见 [第三方声明](THIRD_PARTY_NOTICES.md)。模型权重不包含在仓库中，使用前应阅读上游许可。

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [InternVL3.5](https://huggingface.co/OpenGVLab/InternVL3_5-14B)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [FastAPI](https://github.com/fastapi/fastapi)、[OpenCV](https://github.com/opencv/opencv)
