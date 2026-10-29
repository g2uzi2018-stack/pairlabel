"""An unreadable target is a review state, never a new training category."""
UNKNOWN_CLASS = -1


def unreadable(obj):
    return obj.get('classification_status') == 'unreadable'


def review_class_id(obj):
    return UNKNOWN_CLASS if unreadable(obj) else obj['class_id']


def assign_class(obj, cid, classes, source='human'):
    by_id = {c['id']: c for c in classes}
    if type(cid) is not int or (cid != UNKNOWN_CLASS and cid not in by_id):
        raise ValueError('选择的类别不在本任务类别清单中')
    if cid == UNKNOWN_CLASS:
        # Keep a historical class ID for old readers; all training encoders block
        # unresolved objects, and the viewer explicitly displays the review state.
        if not unreadable(obj):
            obj['classification_previous_class_id'] = obj.get('class_id')
        if obj.get('class_id') not in by_id:
            obj['class_id'] = classes[0]['id']
        obj['classification_status'] = 'unreadable'
        obj['classification_reason'] = '看不清／无法可靠分类'
        obj['color'] = 'unknown'
    else:
        obj['class_id'] = cid
        obj['classification_status'] = 'reviewed'
        obj.pop('classification_reason', None)
        obj.pop('classification_previous_class_id', None)
        color = by_id[cid].get('color')
        if color in ('red', 'black'):
            obj['color'] = color
        else:
            obj.pop('color', None)
    obj['classification_source'] = source


def require_readable(objects):
    if any(unreadable(o) for o in objects):
        raise ValueError('存在看不清／无法可靠分类的目标，整图暂不导出，避免将未知目标当作背景')
