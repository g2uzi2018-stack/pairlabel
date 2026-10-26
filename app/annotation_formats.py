"""Shared export encoders and read-back previews; previews never run inference."""
import json

FORMATS = ('yolo', 'yolo-seg', 'coco')


def has_complete_polygons(objects):
    return all(len(o.get('polygon', [])) >= 3 for o in objects)


def yolo_text(objects, width, height, segmentation=False):
    lines = []
    for o in objects:
        if segmentation:
            values = [v for p in o['polygon'] for v in (p[0] / width, p[1] / height)]
        else:
            x0, y0, x1, y1 = o['bbox']
            values = [(x0+x1)/2/width, (y0+y1)/2/height, (x1-x0)/width, (y1-y0)/height]
        lines.append(str(o['class_id']) + ' ' + ' '.join(f'{v:.7f}' for v in values))
    return '\n'.join(lines) + '\n' if lines else ''


def coco_annotations(objects, image_id=1, first_id=1):
    annotations = []
    for index, o in enumerate(objects):
        x0, y0, x1, y1 = o['bbox']
        poly = o.get('polygon', [])
        area = (x1-x0)*(y1-y0)
        if poly:
            area = abs(sum(poly[k][0]*poly[(k+1) % len(poly)][1] -
                           poly[(k+1) % len(poly)][0]*poly[k][1] for k in range(len(poly))))/2
        annotations.append({'id': first_id+index, 'image_id': image_id,
                            'category_id': o['class_id']+1, 'bbox': [x0, y0, x1-x0, y1-y0],
                            'area': area, 'segmentation': [[v for p in poly for v in p]] if poly else [],
                            'iscrowd': 0})
    return annotations


def format_preview(objects, width, height, fmt, classes):
    if fmt not in FORMATS:
        raise ValueError('不支持的预览格式')
    complete = has_complete_polygons(objects)
    if fmt == 'yolo-seg' and not complete:
        return {'objects': [], 'annotation': '', 'format_eligible': False,
                'format_reason': '存在目标缺少有效轮廓，YOLO 分割会排除整张图片，不输出残缺标签。'}
    if fmt == 'coco':
        payload = {'images': [{'id': 1, 'width': width, 'height': height}],
                   'categories': [{'id': c['id']+1, 'name': c['name'],
                                   'supercategory': c.get('color', 'object')} for c in classes],
                   'annotations': coco_annotations(objects)}
        annotation = json.dumps(payload, ensure_ascii=False, indent=2)
        decoded = json.loads(annotation)
        viewed = []
        for o in decoded['annotations']:
            x, y, w, h = o['bbox']
            flat = o['segmentation'][0] if o['segmentation'] else []
            viewed.append({'class_id': o['category_id']-1, 'bbox': [x, y, x+w, y+h],
                           'polygon': [flat[k:k+2] for k in range(0, len(flat), 2)]})
    else:
        annotation = yolo_text(objects, width, height, fmt == 'yolo-seg')
        viewed = []
        for line in annotation.splitlines():
            tokens = line.split()
            values = list(map(float, tokens[1:]))
            if fmt == 'yolo':
                x, y, w, h = values
                bbox = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
                polygon = []
            else:
                polygon = [[values[k]*width, values[k+1]*height] for k in range(0, len(values), 2)]
                xs, ys = zip(*polygon)
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            viewed.append({'class_id': int(tokens[0]), 'bbox': bbox, 'polygon': polygon})
    return {'objects': viewed, 'annotation': annotation, 'format_eligible': True,
            'format_reason': '', 'has_complete_polygons': complete}
