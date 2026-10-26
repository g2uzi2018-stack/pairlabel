"""Read-only progress from durable results; stages are counts, not time estimates."""

def summarize(items, runs):
    pending = [i['attempt'] for i in items if i['status'] == 'pending']
    attempt = min(pending) if pending else max((i['attempt'] for i in items), default=0)
    members = {i['id'] for i in items if attempt == 0 or i['attempt'] >= attempt}
    members.update(r['item_id'] for r in runs if r['attempt'] == attempt)
    scope = [i for i in items if i['id'] in members]
    total = len(scope)
    result = {'round': attempt + 1, 'total': total}
    for role in ('generator', 'reviewer'):
        selected = [r for r in runs if r['attempt'] == attempt and r['role'] == role and r['item_id'] in members]
        result[role] = {
            'done': len({r['item_id'] for r in selected if r['status'] == 'ok'}),
            'failed': len({r['item_id'] for r in selected if r['status'] != 'ok'}),
            'total': total,
        }
    result['reviewed'] = {
        'done': sum((i['result_attempt'] is not None and i['result_attempt'] >= attempt)
                    or (i['status'] != 'pending' and i['attempt'] >= attempt) for i in scope),
        'total': total,
    }
    result['retry_runs'] = sum(r['attempt'] > 0 and r['status'] == 'ok' for r in runs)
    return result
