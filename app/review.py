"""Shared review classification and label-only edits, preserving geometry/evidence."""
import copy

# The final item can be 'uncertain' even when a model request failed. Historical
# failed calls from earlier attempts do not classify a subsequently resolved item.
FAILED_SQL = "(status='error' OR (status='uncertain' AND EXISTS (SELECT 1 FROM json_each(items.issues) issue WHERE json_extract(issue.value,'$.type')='inference_error')))"


def recognition_failed(item):
    return item['status'] == 'error' or (item['status'] == 'uncertain' and
        any(issue.get('type') == 'inference_error' for issue in (item.get('issues') or [])))


def relabel(result, class_ids, classes, expected=None):
    result = copy.deepcopy(result or {'objects': []})
    objects = result.get('objects', [])
    if not objects:
        raise ValueError('没有可修改的目标，请重试识别，或用高级编辑补充目标')
    if class_ids is None or len(class_ids) != len(objects):
        raise ValueError('类别数量与目标数量不一致，请刷新后重新确认')
    old_ids = [o['class_id'] for o in objects]
    if expected is not None and expected != old_ids:
        raise ValueError('类别已被其他操作更新，请重新打开图片后确认')
    by_id = {c['id']: c for c in classes}
    if any(type(cid) is not int or cid not in by_id for cid in class_ids):
        raise ValueError('选择的类别不在本任务类别清单中')
    changes = []
    for index, (obj, cid) in enumerate(zip(objects, class_ids)):
        if cid == obj['class_id']:
            continue
        changes.append({'index': index, 'before': obj['class_id'], 'after': cid})
        obj['class_id'] = cid
        if by_id[cid].get('color') in ('red', 'black'):
            obj['color'] = by_id[cid]['color']
        else:
            obj.pop('color', None)
    if changes:
        result['human_edited'] = True
    reason = '用户确认本图标签（类别未修改）。'
    if changes:
        reason = '用户确认类别修改：' + '；'.join(
            f"目标 {c['index']+1}：{by_id[c['before']]['label']} → {by_id[c['after']]['label']}" for c in changes)
    return result, changes, reason
