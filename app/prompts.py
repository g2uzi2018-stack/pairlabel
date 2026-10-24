"""Shared, dependency-free annotation instructions for independent readers."""
from .profiles import is_xiangqi

BOX_RULES='bbox 应紧密包住可见目标的完整实体外缘，不是只包文字；使用相对于当前输入整张图片的 [左,上,右,下]，归一化到 0–1000，满足 0 <= 左 < 右 <= 1000、0 <= 上 < 下 <= 1000。裁剪或分块输入的坐标相对于该输入图片，由程序换算回原图。'
SCORE_RULES='confidence 是你自己的确定程度，必须为 0 到 1 之间、最多两位小数的数值，例如 0.95；禁止 1–5 评分或百分数，它不是校准概率。不确定时降低分数，不要编造。'
TEXT_RULES='text 仅填写目标上实际可读的文字，没有可读文字时填空字符串。物体识别依据可见外观，不要求先读出文字；无文字本身不代表无法识别。'
XIANGQI_RULES='中国象棋必须读清文字并区分红黑，注意炮/砲、车/車、马/馬、相/象、仕/士、帅/将、兵/卒。红黑指棋子文字或阵营颜色，不是背景颜色。部分棋子可能旋转或倒置。不要凭棋局规则或初始摆法推测被遮挡或不存在的棋子。每个棋子独立填写 color：红色文字 red，黑/蓝/绿色文字归 black，无法看清颜色 unknown；不要只用 class_id 代替颜色判断。'

def _task(c):
 classes='\n'.join(f'{x["id"]}: {x["label"]} ({x["name"]})' for x in c['classes'])
 instructions=c.get('annotation_instructions','')
 return f'类别（class_id 使用下列整数）：\n{classes}\n用户标注要求（在上述类别、坐标和 JSON 输出规范内执行）：\n{instructions or "无额外要求"}'

def build_detection_prompt(c,attempt):
 parts=['你是严谨的图像数据标注员。独立观察这张完整图片，检测所有属于下列类别的可见目标，不参考其他模型的输出。仅依据图像证据，不推测遮挡后或不存在的目标。',_task(c),
  '对每个可见目标只输出一次，不要省略重复类别。',BOX_RULES,SCORE_RULES]
 if is_xiangqi(c):
  parts.extend([XIANGQI_RULES,'text 写实际识别到的文字。疑似棋子但背面、遮挡或字不清无法分类时，不要猜类别，写入 unknown_regions。'])
 else:
  parts.extend([TEXT_RULES,'看见疑似目标但因遮挡、模糊或类别不明无法可靠分类时，不要猜类别，写入 unknown_regions。'])
 parts.append('只输出一个 JSON 对象，字段为 objects、unknown_regions、image_quality。objects 每项包含 class_id、bbox、confidence、text'+('、color' if is_xiangqi(c) else '')+'；unknown_regions 每项包含 bbox 和 reason，遵守相同框规范。没有目标时 objects=[]，没有疑似未知目标时 unknown_regions=[]。image_quality 用字符串简述遮挡、模糊情况。不输出 Markdown、解释或其他字段。')
 if attempt:parts.append('这是独立重新观察：按从上到下、从左到右逐行扫描，检查边缘小目标、遮挡和容易混淆的外观。')
 return '\n'.join(parts)

def build_crop_prompt(c,count):
 parts=[f'独立识别拼图中 PATCH 1 到 PATCH {count} 的中心实体，每个 PATCH 输出一次。只判断中心候选物，不识别编号文字，不参考其他 PATCH 或其他模型的判断。',_task(c),
  '定位沿用像素候选在原图中的 bbox，不另行生成定位；原图候选必须遵守完整框规范：'+BOX_RULES,SCORE_RULES]
 if is_xiangqi(c):
  parts.extend([XIANGQI_RULES,'若是可读中国象棋棋子，kind="piece"，读出文字 text 并填写 class_id。若是界面按钮、硬币、背景等非棋子，kind="not_piece"，class_id=-1。若是棋子但背面或文字看不清，kind="unreadable"，class_id=-1 并说明 reason。非棋子或看不清颜色时 color="unknown"。'])
 else:
  parts.extend([TEXT_RULES,'若能可靠归入配置类别，kind="object" 并填写 class_id。明确不属于配置类别时 kind="not_target"，class_id=-1。疑似目标但无法可靠分类时 kind="unreadable"，class_id=-1 并说明 reason。'])
 parts.append('只输出 JSON 对象 {"items":[...]}，按 PATCH 编号填写每项的 patch、kind、class_id、text、confidence、reason'+('、color' if is_xiangqi(c) else '')+'，不输出 bbox 或其他字段。没有额外原因时 reason=""。不输出 Markdown 或解释。')
 return '\n'.join(parts)
