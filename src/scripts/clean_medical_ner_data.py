"""Apply a source-bound, explicit review plan; never infer decisions from prose."""
import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent.parent
LABELS = ('dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep')
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ENTITY_KEYS = ('start_idx', 'end_idx', 'type', 'entity')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_review(path):
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() != '.xlsx':
        raise ValueError('Review must be CSV or XLSX')
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//s:t', NS))
                       for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        tables = []
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            rid = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            target = rels[rid]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            rows = []
            for row in ET.fromstring(z.read(target)).findall('s:sheetData/s:row', NS):
                values = {}
                for cell in row.findall('s:c', NS):
                    col = ''.join(c for c in cell.get('r') if c.isalpha())
                    if cell.find('s:f', NS) is not None:
                        raise ValueError(f'Review formulas unsupported: {cell.get("r")}')
                    v = cell.find('s:v', NS)
                    value = v.text or '' if v is not None else ''
                    if cell.get('t') == 's':
                        value = strings[int(value)]
                    elif cell.get('t') == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.findall('s:is//s:t', NS))
                    elif cell.get('t') == 'b':
                        value = 'True' if value == '1' else 'False'
                    values[col] = value
                if any(values.values()):
                    rows.append(values)
            if rows and 'group_id' in rows[0].values():
                header = rows[0]
                if len(set(header.values())) != len(header):
                    raise ValueError('Duplicate review headers')
                tables.append([{name: row.get(col, '') for col, name in header.items()} for row in rows[1:]])
        if len(tables) != 1:
            raise ValueError('Expected exactly one review table')
        return tables[0]


def signature(entities):
    return Counter(tuple(e[k] for k in ENTITY_KEYS) for e in entities)


def valid_entity(text, e):
    if not isinstance(e, dict) or any(k not in e for k in ENTITY_KEYS):
        return False
    s, t = e['start_idx'], e['end_idx']
    return (type(s) is int and type(t) is int and 0 <= s <= t < len(text)
            and e['type'] in LABELS and isinstance(e['entity'], str)
            and text[s:t + 1] == e['entity'])


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data_raw/CMeEE_train.json')
    parser.add_argument('--review', type=Path, default=ROOT / 'docs/duplicate_annotation_review_checked.csv.xlsx')
    parser.add_argument('--plan', type=Path, default=ROOT / 'data_cleaned/cleaning_plan.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data_cleaned')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw_dir = (ROOT / 'data_raw').resolve()
    if output == raw_dir or raw_dir in output.parents:
        raise ValueError('Output must not be in data_raw')
    # Snapshot every project raw JSON, the selected input, review, and plan.
    protected = set(raw_dir.glob('*.json')) | {args.input.resolve(), args.review.resolve(), args.plan.resolve()}
    output_names = ['CMeEE_train_cleaned.json', 'cleaning_log.csv', 'cleaning_summary.md',
                    'remaining_review_items.csv', 'validation_report.json', 'record_index_map.csv']
    if any((output / name).resolve() in protected for name in output_names):
        raise ValueError('An output would overwrite a protected input')
    before_hashes = {str(p): digest(p) for p in protected}
    output.mkdir(parents=True, exist_ok=True)
    logs, remaining = [], []
    try:
        source = json.loads(args.input.read_text(encoding='utf-8-sig'))
        if not isinstance(source, list):
            raise ValueError('Input must be JSON list')
        cleaned = copy.deepcopy(source)
        rows = read_review(args.review)
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        if digest(args.input) != plan['source_sha256'] or digest(args.review) != plan['review_sha256']:
            raise ValueError('Source/review changed: re-inspect and update explicit plan; no automatic prose interpretation')
        by_group = {}
        for row in rows:
            gid = int(row['group_id'])
            if gid in by_group:
                raise ValueError(f'Duplicate group {gid}')
            by_group[gid] = row
        if set(by_group) != set(map(int, plan['groups'])):
            raise ValueError('Plan/review group coverage differs')
        text_indices = defaultdict(list)
        for i, r in enumerate(source):
            if not isinstance(r, dict) or not isinstance(r.get('text'), str) or not isinstance(r.get('entities'), list):
                raise ValueError(f'Invalid input structure: record {i}')
            text_indices[r['text']].append(i)
        actual_duplicates = {t: ids for t, ids in text_indices.items() if len(ids) > 1}
        if len(rows) != len(actual_duplicates):
            raise ValueError('Review does not cover all source duplicate groups')
        excluded = set()
        changed = set()
        errors = []
        counters = Counter()

        def log(i, gid, action, reason, old, new, status):
            logs.append(dict(record_index=i, group_id=gid, text=source[i]['text'],
                             original_entities=old, cleaned_entities=new, action=action,
                             reason=reason, review_status=status))

        def wait(gid, row, reason, action):
            remaining.append(dict(group_id=gid, record_indices=row['record_indices'], text=row['text'],
                                  manual_review_status=row['manual_review_status'],
                                  processing_status='manual_intervention_required', reason=reason,
                                  review_notes=row['review_notes'], cleaning_recommendation=row.get('cleaning_recommendation', ''),
                                  action=action))

        for gid, row in by_group.items():
            rule = plan['groups'][str(gid)]
            ids = json.loads(row['record_indices'])
            try:
                if (not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids)
                        or any(type(i) is not int or not 0 <= i < len(source) for i in ids)):
                    raise ValueError('Invalid/out-of-range record_indices')
                if any(source[i]['text'] != row['text'] for i in ids):
                    raise ValueError('Review text differs from source')
                if sorted(ids) != actual_duplicates.get(row['text']):
                    raise ValueError('Review does not contain ALL records for this text')
                counts = json.loads(row['entity_counts'])
                details = json.loads(row['all_record_entities'])
                for pos, i in enumerate(ids):
                    es = source[i]['entities']
                    if len(es) != counts[str(i)] or not all(valid_entity(source[i]['text'], e) for e in es):
                        raise ValueError(f'Invalid/stale original annotations for record {i}')
                    if str(i) in details and signature(es) != Counter(map(tuple, details[str(i)])):
                        raise ValueError(f'Stale all_record_entities: {i}')
                    if pos < 2 and row.get(('record_a_entities', 'record_b_entities')[pos]):
                        if signature(es) != Counter(map(tuple, json.loads(row[('record_a_entities', 'record_b_entities')[pos]]))):
                            raise ValueError(f'Stale record entity cell: {i}')
                consistent = all(signature(source[i]['entities']) == signature(source[ids[0]]['entities']) for i in ids)
                if consistent != (str(row['annotations_consistent']).lower() in ('true', '1')):
                    raise ValueError('Consistency flag differs from actual source')
                if consistent:
                    counters['consistent_groups_preserved'] += 1
                    continue
                if rule['action'] == 'exclude':
                    if row['manual_review_status'] != 'pending':
                        raise ValueError('Exclusion rule conflicts with current status')
                    for i in ids:
                        excluded.add(i)
                        log(i, gid, 'exclude_disputed_record', rule['reason'] + '\n' + row['review_notes'],
                            source[i]['entities'], [], row['manual_review_status'])
                    counters['excluded_groups'] += 1
                    wait(gid, row, rule['reason'], 'temporarily_excluded')
                    continue
                if row['manual_review_status'] != 'reviewed' or rule['action'] != 'edit':
                    wait(gid, row, 'No explicit executable reviewed rule', 'unchanged')
                    continue
                removals = rule['remove']
                additions = rule['ensure']
                pool = signature([e for i in ids for e in source[i]['entities']])
                if any(tuple(e) not in pool for e in removals + additions):
                    raise ValueError('Planned entity is absent from reviewed source annotations')
                # Stage the entire group before mutating it: failure cannot leave half a group edited.
                staged = {}
                for i in ids:
                    es = copy.deepcopy(cleaned[i]['entities'])
                    for target in removals:
                        es = [e for e in es if tuple(e[k] for k in ENTITY_KEYS) != tuple(target)]
                    for target in additions:
                        if tuple(target) not in signature(es):
                            es.append(dict(zip(ENTITY_KEYS, target)))
                    if not all(valid_entity(source[i]['text'], e) for e in es):
                        raise ValueError(f'Invalid proposed annotation: {i}')
                    staged[i] = es
                for i, es in staged.items():
                    old = cleaned[i]['entities']
                    removed = list((signature(old) - signature(es)).elements())
                    added = list((signature(es) - signature(old)).elements())
                    if removed or added:
                        counters['removed_review_entities'] += len(removed)
                        counters['added_review_entities'] += len(added)
                        changed.add(i)
                        log(i, gid, 'apply_explicit_review', row['review_notes'] +
                            '\n仅落实明确实体操作；未提及实体按原记录保留。差异：' +
                            json.dumps({'removed': removed, 'added': added}, ensure_ascii=False),
                            copy.deepcopy(old), copy.deepcopy(es), row['manual_review_status'])
                        cleaned[i]['entities'] = es
                if not all(signature(cleaned[i]['entities']) == signature(cleaned[ids[0]]['entities']) for i in ids):
                    unresolved = [dict(record_index=i, entities=cleaned[i]['entities']) for i in ids]
                    reason = rule.get('remaining_reason', 'Explicit decisions do not fully resolve all annotation differences')
                    wait(gid, row, reason + '\n当前保留标注：' + json.dumps(unresolved, ensure_ascii=False),
                         'explicit_edits_applied_remaining_annotations_preserved')
                    for i in ids:
                        log(i, gid, 'manual_intervention_required', reason, copy.deepcopy(cleaned[i]['entities']),
                            copy.deepcopy(cleaned[i]['entities']), row['manual_review_status'])
                else:
                    counters['unified_groups'] += 1
            except (ValueError, KeyError, TypeError) as exc:
                reason = str(exc)
                errors.append({'group_id': gid, 'error': reason})
                wait(gid, row, reason, 'group_stopped_unchanged')
                # Avoid potentially invalid indices in failure logging.
                logs.append(dict(record_index=row.get('record_indices', ''), group_id=gid, text=row.get('text', ''),
                                 original_entities=row.get('all_record_entities', ''), cleaned_entities='',
                                 action='group_validation_failed', reason=reason, review_status=row.get('manual_review_status', '')))
        for anomaly in plan['remove_invalid_entities']:
            i = anomaly['record_index']
            target = tuple(anomaly['entity'])
            matches = [e for e in cleaned[i]['entities'] if tuple(e[k] for k in ENTITY_KEYS) == target]
            if len(matches) != 1 or valid_entity(source[i]['text'], matches[0]):
                raise ValueError(f'Expected unique invalid entity not found: {i}')
            old = copy.deepcopy(cleaned[i]['entities'])
            cleaned[i]['entities'] = [e for e in old if tuple(e[k] for k in ENTITY_KEYS) != target]
            counters['invalid_entities_removed'] += 1
            changed.add(i)
            log(i, '', 'remove_empty_reversed_entity', '用户明确授权：移出有效训练标签，保留原字段和文本记录。',
                old, copy.deepcopy(cleaned[i]['entities']), 'user_authorized')
        kept_ids = [i for i in range(len(source)) if i not in excluded]
        result = [cleaned[i] for i in kept_ids]
        for i in kept_ids:
            if cleaned[i]['text'] != source[i]['text'] or any(not valid_entity(cleaned[i]['text'], e) for e in cleaned[i]['entities']):
                raise ValueError(f'Final validation failed: {i}')
            # Text and any unrecognized record fields must remain unchanged.
            if {k: v for k, v in cleaned[i].items() if k != 'entities'} != {k: v for k, v in source[i].items() if k != 'entities'}:
                raise ValueError(f'Unexpected record field modification: {i}')
        log_ids = {r['record_index'] for r in logs if type(r['record_index']) is int}
        actual_changed = {i for i in kept_ids if source[i] != cleaned[i]}
        if not actual_changed <= log_ids or not excluded <= log_ids:
            raise ValueError('Unlogged modifications/exclusions')
        if errors:
            raise ValueError('Group validation failures; see failure logs; cleaned dataset not published')
        after_hashes = {p: digest(Path(p)) for p in before_hashes}
        if before_hashes != after_hashes:
            raise ValueError('Protected input SHA256 changed')
        counts_before = Counter(e['type'] for r in source for e in r['entities'])
        counts_after = Counter(e['type'] for r in result for e in r['entities'])
        validation = dict(passed=True, records_before=len(source), records_after=len(result),
                          entities_before=sum(counts_before.values()), entities_after=sum(counts_after.values()),
                          labels_before=dict(counts_before), labels_after=dict(counts_after),
                          modified_records=len(actual_changed), excluded_records=len(excluded),
                          remaining_groups=len(remaining), counters=dict(counters),
                          protected_sha256_before=before_hashes, protected_sha256_after=after_hashes,
                          group_validation_errors=errors, all_changes_logged=True,
                          note='结构和追溯校验通过不代表所有语义争议已解决；部分 reviewed 组仍有未裁定实体差异。')
        dataset_path = output / 'CMeEE_train_cleaned.json'
        dataset_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if json.loads(dataset_path.read_text(encoding='utf-8')) != result:
            raise ValueError('Saved JSON round-trip differs')
        validation['cleaned_sha256'] = digest(dataset_path)
        write_csv(output / 'record_index_map.csv', [dict(cleaned_record_index=j, original_record_index=i) for j, i in enumerate(kept_ids)],
                  ['cleaned_record_index', 'original_record_index'])
        write_csv(output / 'cleaning_log.csv', logs, ['record_index', 'group_id', 'text', 'original_entities', 'cleaned_entities', 'action', 'reason', 'review_status'])
        write_csv(output / 'remaining_review_items.csv', remaining, ['group_id', 'record_indices', 'text', 'manual_review_status',
                  'processing_status', 'reason', 'review_notes', 'cleaning_recommendation', 'action'])
        (output / 'validation_report.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary = ['# 医疗 NER 清洗副本说明', '', f'输入：`{args.input.resolve()}`', f'人工记录：`{args.review.resolve()}`',
                   f'可执行计划：`{args.plan.resolve()}`（绑定输入与人工记录哈希；不自动解释任意自然语言备注）', '',
                   f'- 样本数：{len(source)} → {len(result)}。',
                   f'- 实体数：{sum(counts_before.values())} → {sum(counts_after.values())}。',
                   f'- 修改记录：{len(actual_changed)}；人工复核操作新增实体 {counters["added_review_entities"]} 条、移除旧实体 {counters["removed_review_entities"]} 条。替换按移除旧条目和新增新条目分别计数，不重复声称为独立纠错数量。',
                   f'- 暂时排除 {len(excluded)} 条记录、{counters["excluded_groups"]} 组；只排除训练副本，不能解释为所有实体均错误。',
                   f'- 额外移除异常实体 {counters["invalid_entities_removed"]} 条，所在文本保留。',
                   f'- 一致重复组保留 {counters["consistent_groups_preserved"]} 组，未去重；完全统一 {counters["unified_groups"]} 个原不一致组。',
                   f'- 仍需人工介入 {len(remaining)} 组：' + '、'.join(str(r['group_id']) for r in remaining) + '。',
                   '- 第 9、11、41 组仅执行明确操作，未提及的差异保持各自原标注；未强行统一。清洗版并非所有标注都已裁定。',
                   '- 第 25 组建议中“保留一份一致标注”未视为明确的文本去重授权，保留全部记录。',
                   '- dev/test 不改动、不生成清洗版；test 空 entities 不作为真实负标签或用于计算 P/R/F1。',
                   '- JSON 回读、字段、九类标签、闭区间索引、实体切片、修改日志覆盖和源文件 SHA-256 检查均通过。', '',
                   '| 标签 | 清洗前 | 清洗后 | 变化 |', '| --- | ---: | ---: | ---: |']
        summary += [f'| {label} | {counts_before[label]} | {counts_after[label]} | {counts_after[label]-counts_before[label]:+d} |' for label in LABELS]
        summary += ['', '## 待人工介入事项'] + [f'- 第 {r["group_id"]} 组：{r["reason"]}' for r in remaining]
        summary += ['', '## 输入完整性', '以下原始 JSON 和人工记录运行前后 SHA-256 完全一致：']
        summary += [f'- `{p}`：`{value}`' for p, value in sorted(before_hashes.items())]
        summary += ['', '原始下标均为零基。清洗版因排除记录而改变行序号，`record_index_map.csv` 保存清洗行号到原始行号的对应关系。',
                    '重叠与嵌套未因其存在而移除；原始 JSON、人工工作簿、已有报告均不修改。未转换 BIO。',
                    '复现：在项目根目录运行 `"C:\\Users\\home\\anaconda3\\python.exe" -X utf8 scripts\\clean_medical_ner_data.py`。',
                    '可用 `--input`、`--review`、`--plan`、`--output-dir` 指定路径；不同数据/新备注必须先更新并核实计划，哈希不匹配会停止。',
                    '重跑会更新输出文件；不要在自动生成日志中直接写入唯一的人工复核记录。']
        (output / 'cleaning_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
        print(json.dumps({k: validation[k] for k in ('records_before','records_after','entities_before','entities_after','modified_records','excluded_records','remaining_groups','counters')}, ensure_ascii=False, indent=2))
        print('Validation PASS; original and review input hashes unchanged.')
    except Exception as exc:
        write_csv(output / 'cleaning_failure_log.csv', logs + [dict(record_index='', group_id='', text='', original_entities='',
                  cleaned_entities='', action='run_failed', reason=str(exc), review_status='manual_intervention_required')],
                  ['record_index', 'group_id', 'text', 'original_entities', 'cleaned_entities', 'action', 'reason', 'review_status'])
        raise


if __name__ == '__main__':
    import sys
    if '--final' in sys.argv:
        from src.scripts.finalize_medical_ner_data import main as final_main
        sys.argv.remove('--final')
        final_main()
    else:
        main()
