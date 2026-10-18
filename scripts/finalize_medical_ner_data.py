"""Exclude ALL unresolved groups by original index, preserving the previous revision."""
import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from clean_medical_ner_data import ROOT, LABELS, valid_entity, digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return buffer.getvalue().encode('utf-8-sig')


def validate_records(records, label):
    require(isinstance(records, list), f'{label}: expected list')
    counts = Counter()
    for i, record in enumerate(records):
        require(isinstance(record, dict) and isinstance(record.get('text'), str)
                and isinstance(record.get('entities'), list), f'{label}: invalid record {i}')
        for entity in record['entities']:
            require(valid_entity(record['text'], entity), f'{label}: invalid entity in record {i}: {entity}')
            counts[entity['type']] += 1
    return counts


def build(args):
    previous_dir = args.previous_dir.resolve()
    docs = args.docs_dir.resolve()
    required = {name: previous_dir / name for name in (
        'CMeEE_train_cleaned.json', 'remaining_review_items.csv', 'cleaning_plan.json',
        'cleaning_log.csv', 'record_index_map.csv', 'validation_report.json')}
    source_path = args.input.resolve()
    for p in [source_path, *required.values()]:
        require(p.is_file(), f'Missing necessary file: {p}')
    protected = set(required.values()) | {source_path} | set((ROOT / 'data_raw').glob('*.json'))
    # Read/hash all seven specified inputs; do not trust old validation alone.
    hashes_before = {str(p): digest(p) for p in sorted(protected)}
    source = read_json(source_path)
    previous = read_json(required['CMeEE_train_cleaned.json'])
    pending = read_csv(required['remaining_review_items.csv'])
    plan = read_json(required['cleaning_plan.json'])
    logs = read_csv(required['cleaning_log.csv'])
    mapping = read_csv(required['record_index_map.csv'])
    old_validation = read_json(required['validation_report.json'])
    require(isinstance(source, list), 'Original training data must be list')
    require(digest(source_path) == plan['source_sha256'], 'Original training hash differs from prior plan')
    require(old_validation['cleaned_sha256'] == digest(required['CMeEE_train_cleaned.json']),
            'Previous cleaned file differs from its recorded hash')
    previous_counts = validate_records(previous, 'previous')
    original_counts = Counter()
    text_indices = defaultdict(list)
    for i, record in enumerate(source):
        require(isinstance(record, dict) and isinstance(record.get('text'), str)
                and isinstance(record.get('entities'), list), f'Invalid original record {i}')
        text_indices[record['text']].append(i)
        for entity in record['entities']:
            require(isinstance(entity, dict) and entity.get('type') in LABELS,
                    f'Invalid original entity structure/type {i}')
            original_counts[entity['type']] += 1

    # Replay every historical change, including removals, to prove the old dataset's provenance.
    replay = copy.deepcopy(source)
    old_excluded = set()
    for entry in logs:
        i = int(entry['record_index'])
        require(0 <= i < len(source), 'Historical log index out of range')
        require(entry['text'] == source[i]['text'], f'Historical log text mismatch: {i}')
        before = json.loads(entry['original_entities'])
        after = json.loads(entry['cleaned_entities'])
        require(replay[i]['entities'] == before, f'Historical log chain mismatch: {i}')
        action = entry['action']
        if action == 'exclude_disputed_record':
            require(i not in old_excluded and after == [], f'Duplicate/invalid historical exclusion: {i}')
            old_excluded.add(i)
            gid = str(int(entry['group_id']))
            require(plan['groups'][gid]['action'] == 'exclude', f'Exclusion not in prior plan: {i}')
        elif action in ('apply_explicit_review', 'remove_empty_reversed_entity'):
            require(i not in old_excluded, f'Historical modification of excluded record: {i}')
            if action == 'apply_explicit_review':
                rule = plan['groups'][str(int(entry['group_id']))]
                require(rule['action'] == 'edit' and entry['review_status'] == 'reviewed', 'Unapproved historical change')
                expected = copy.deepcopy(before)
                for target in rule['remove']:
                    expected = [e for e in expected if [e[k] for k in ('start_idx','end_idx','type','entity')] != target]
                for target in rule['ensure']:
                    e = dict(zip(('start_idx','end_idx','type','entity'), target))
                    if not any(all(item.get(k) == v for k,v in e.items()) for item in expected):
                        expected.append(e)
                require(expected == after, f'Historical edits deviate from explicit plan: {i}')
            else:
                targets = [r['entity'] for r in plan['remove_invalid_entities'] if r['record_index'] == i]
                require(len(targets) == 1, 'Unknown historical anomaly removal')
                expected = [e for e in before if [e[k] for k in ('start_idx','end_idx','type','entity')] != targets[0]]
                require(expected == after and len(before)-len(after) == 1, 'Invalid anomaly removal log')
            replay[i]['entities'] = after
        elif action == 'manual_intervention_required':
            require(before == after, 'Intervention log must not change data')
        else:
            raise ValueError(f'Unknown historical action: {action}')

    require(len(mapping) == len(previous), 'Previous map length mismatch')
    old_ids = []
    for j, row in enumerate(mapping):
        i = int(row['original_record_index'])
        require(int(row['cleaned_record_index']) == j and 0 <= i < len(source), f'Invalid previous mapping row {j}')
        require(previous[j] == replay[i], f'Previous dataset cannot be replayed from logs: {i}')
        old_ids.append(i)
    require(len(set(old_ids)) == len(old_ids), 'Duplicate original IDs in previous map')
    require(set(old_ids) == set(range(len(source))) - old_excluded, 'Previous mapping/exclusion coverage differs')

    groups, unresolved_ids, removal_rows = set(), set(), []
    old_positions = {i: j for j,i in enumerate(old_ids)}
    for row in pending:
        gid = int(row['group_id'])
        require(gid not in groups, f'Duplicate unresolved group: {gid}')
        groups.add(gid)
        ids = json.loads(row['record_indices'])
        require(isinstance(ids, list) and len(ids) >= 2 and all(type(i) is int for i in ids), f'Invalid pending indices: {gid}')
        require(len(set(ids)) == len(ids) and not unresolved_ids.intersection(ids), f'Repeated pending record: {gid}')
        require(all(0 <= i < len(source) for i in ids), f'Pending indices out of range: {gid}')
        require(all(source[i]['text'] == row['text'] for i in ids), f'Pending text mismatch: {gid}')
        require(sorted(ids) == text_indices[row['text']], f'Pending group omits original records: {gid}')
        unresolved_ids.update(ids)
        for i in ids:
            j = old_positions.get(i)
            removal_rows.append(dict(group_id=gid, original_record_index=i,
                                     previous_cleaned_record_index=j if j is not None else '', text=source[i]['text'],
                                     original_entity_count=len(source[i]['entities']),
                                     previous_entity_count=len(previous[j]['entities']) if j is not None else '',
                                     original_entities=source[i]['entities'],
                                     previous_entities=previous[j]['entities'] if j is not None else None,
                                     exclusion_status='already_excluded' if i in old_excluded else 'newly_excluded',
                                     review_status=row['manual_review_status'],
                                     reason='用户决定：剩余争议组整组排除，不再逐实体裁定；不代表所有实体均错误。',
                                     prior_dispute_reason=row['reason']))
    require(groups, 'Unresolved review list is empty')
    require(old_excluded <= unresolved_ids, 'Previously excluded records are missing from unresolved list; stop for reconciliation')
    union_excluded = old_excluded | unresolved_ids
    final_ids = [i for i in old_ids if i not in union_excluded]
    final = [previous[old_positions[i]] for i in final_ids]
    final_counts = validate_records(final, 'final')
    require(len(set(final_ids)) == len(final_ids) and set(final_ids).isdisjoint(unresolved_ids | old_excluded),
            'Final mapping duplicate or disputed ID remains')
    require(set(final_ids) == set(range(len(source))) - union_excluded, 'Final index coverage incomplete')
    require(all(final[j] == previous[old_positions[i]] and final[j] == replay[i] for j,i in enumerate(final_ids)),
            'Unexpected new annotation change')
    map_rows = [dict(final_record_index=j, original_record_index=i, previous_cleaned_record_index=old_positions[i])
                for j,i in enumerate(final_ids)]
    data_bytes = (json.dumps(final, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    require(json.loads(data_bytes) == final, 'Final JSON round-trip mismatch')
    map_bytes = csv_bytes(map_rows, ['final_record_index','original_record_index','previous_cleaned_record_index'])
    require(len(list(csv.DictReader(io.StringIO(map_bytes.decode('utf-8-sig'))))) == len(final), 'Final CSV round-trip mismatch')
    source_hash = hashes_before[str(source_path)]
    final_hash = hashlib.sha256(data_bytes).hexdigest()
    paths = [previous_dir / 'CMeEE_train_cleaned_final.json', docs / 'record_index_map_final.csv',
             docs / 'removed_unresolved_groups.csv', docs / 'final_cleaning_summary.md', docs / 'final_validation_report.json']
    require(not any(p.resolve() in protected or source_path.parent in p.resolve().parents for p in paths), 'Output would overwrite protected input')
    new_count = len(unresolved_ids - old_excluded)
    expected_groups = {4,5,9,11,13,18,30,41}  # Compare/report only; never selects record IDs.
    difference_note = ('与用户提供的8组编号一致。' if groups == expected_groups else
                       f'与用户提供组号不同：文件实际为 {sorted(groups)}；以文件为准。')
    report = dict(passed=True, groups=sorted(groups), group_id_comparison=difference_note,
                  original_records=len(source), previous_records=len(previous), final_records=len(final),
                  original_entities=sum(original_counts.values()), previous_entities=sum(previous_counts.values()),
                  final_entities=sum(final_counts.values()), previously_excluded=len(old_excluded),
                  newly_excluded=new_count, total_excluded=len(union_excluded), excluded_original_indices=sorted(union_excluded),
                  original_labels={k:original_counts[k] for k in LABELS},
                  previous_labels={k:previous_counts[k] for k in LABELS}, final_labels={k:final_counts[k] for k in LABELS},
                  new_annotation_changes=0, historical_changes_replayed=True,
                  checks=['JSON round trip','record structure','nine label types','inclusive offsets and exact entity slices',
                          'unique in-range full index mapping','all disputed and previously excluded indices absent',
                          'historical edits match plan and log','final annotations unchanged from previous retained records',
                          'all source and historical files SHA256 unchanged'],
                  protected_sha256=hashes_before, final_sha256=final_hash)
    summary = ['# CMeEE 最终训练集清洗总结', '',
               '## 本次决定与规则',
               '按 `data_cleaned/remaining_review_items.csv` 中全部剩余组的原始零基记录索引排除整组记录，不再裁定实体。',
               f'实际组号：{sorted(groups)}。{difference_note}',
               f'涉及 {len(unresolved_ids)} 条原始记录：{len(old_excluded)} 条已排除，新增排除 {new_count} 条；最终累计排除 {len(union_excluded)} 条。',
               '从固定的上一版清洗集及原始索引映射筛选，不在删除后的行号上再次删除。保留记录的全部字段和标注与上一版完全一致。',
               '原始训练集、dev/test、上一版清洗文件和人工记录均保留。test 空列表不是负样本金标；未转换 BIO、未训练模型。', '',
               '## 数量统计', '| 数据版本 | 样本数 | 实体数 |', '| --- | ---: | ---: |',
               f'| 原始训练集 | {len(source)} | {sum(original_counts.values())} |',
               f'| 上一版清洗集 | {len(previous)} | {sum(previous_counts.values())} |',
               f'| 最终训练集 | {len(final)} | {sum(final_counts.values())} |', '',
               '| 类型 | 原始 | 上一版 | 最终 | 本次变化 |', '| --- | ---: | ---: | ---: | ---: |']
    summary += [f'| {k} | {original_counts[k]} | {previous_counts[k]} | {final_counts[k]} | {final_counts[k]-previous_counts[k]:+d} |' for k in LABELS]
    summary += ['', '## 验证与追溯',
                '重新检查全部最终实体，不仅引用旧 validation_report；历史清洗日志逐项重放，并对照既定计划验证标注操作。',
                '最终 JSON 回读、字段结构、九类标签、闭区间索引及文本切片通过；索引映射无重复、无越界且逐条匹配文本。',
                '争议索引及此前排除索引均不在最终映射中；无新增实体标注变更。被排除记录明细见 `removed_unresolved_groups.csv`。',
                '本入口每次独立构建两遍并逐字节比较；写盘后再次回读验证。可连续运行下方命令两次比较输出哈希。',
                f'最终 JSON SHA-256：`{final_hash}`', f'原始训练集 SHA-256：`{source_hash}`（运行前后不变）。', '',
                '## 文件与复现',
                '`data_cleaned/CMeEE_train_cleaned_final.json`：最终训练数据。',
                '`docs/record_index_map_final.csv`：最终、原始和上一版行号对应，均为零基。',
                '`docs/removed_unresolved_groups.csv`：完整原始实体、上一版实体、实体数、组号、排除状态及原因。',
                '`docs/final_validation_report.json`：重新计算的统计、校验与输入哈希。',
                '项目根目录执行：', '```bat',
                '"C:\\Users\\home\\anaconda3\\python.exe" -X utf8 -B scripts\\clean_medical_ner_data.py --final', '```',
                '可用 `--input`、`--previous-dir`、`--docs-dir` 指定路径；`--final` 不重跑或覆盖上一版清洗。',
                '本次创建前检查过所有最终输出均不存在；后续仅允许重写内容完全相同的结果，若现有结果不同则停止。', '',
                '## 限制',
                '争议通过整组排除解决训练准入，不代表医学语义已裁定或所有实体均错误；未开展新的标注审定。',
                '第9、11、41组此前部分修订随整条记录一起排除；其余已执行修订保留。',
                '未对标注一致的重复文本去重；数据许可及版本信息沿用现有来源文档。',
                '## 受保护输入 SHA-256']
    summary += [f'- `{p}`：`{sha}`' for p,sha in sorted(hashes_before.items())]
    payloads = {
        paths[0]: data_bytes, paths[1]: map_bytes,
        paths[2]: csv_bytes(removal_rows, ['group_id','original_record_index','previous_cleaned_record_index','text',
            'original_entity_count','previous_entity_count','original_entities','previous_entities','exclusion_status',
            'review_status','reason','prior_dispute_reason']),
        paths[3]: ('\n'.join(summary) + '\n').encode('utf-8'),
        paths[4]: (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')}
    require(all(digest(Path(p)) == sha for p,sha in hashes_before.items()), 'Protected input changed during build')
    return payloads, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data_raw/CMeEE_train.json')
    parser.add_argument('--previous-dir', type=Path, default=ROOT/'data_cleaned')
    parser.add_argument('--docs-dir', type=Path, default=ROOT/'docs')
    args = parser.parse_args()
    payloads, report = build(args)
    repeated, repeated_report = build(args)
    require(payloads == repeated and report == repeated_report, 'Repeated independent builds differ')
    # Preflight every existing result before writing any file: protect historical/user edits.
    for path, content in payloads.items():
        require(not path.exists() or path.read_bytes() == content, f'Existing output differs; refuse overwrite: {path}')
    for path, content in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    require(read_json(args.previous_dir/'CMeEE_train_cleaned_final.json') == json.loads(next(iter(payloads.values()))),
            'Saved final JSON differs')
    for path,content in payloads.items():
        require(path.read_bytes() == content, f'Saved output differs: {path}')
    require(all(digest(Path(p)) == sha for p,sha in report['protected_sha256'].items()), 'Protected input changed after writing')
    print('排除组：', report['groups'])
    print(f"争议记录累计排除 {report['total_excluded']} 条（此前 {report['previously_excluded']}，本次新增 {report['newly_excluded']}）。")
    print(f"最终样本 {report['final_records']}，实体 {report['final_entities']}；验证通过，独立重复构建一致，原始数据不变。")
    print('最终 SHA256：', report['final_sha256'])
    for path in payloads:
        print('输出：', path)


if __name__ == '__main__':
    main()
