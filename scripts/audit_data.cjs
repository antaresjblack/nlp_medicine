// Read-only audit of source data. Run: node scripts/audit_data.cjs
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const report = {offset_convention: 'Unicode code points; source end_idx inclusive', files: {}, cross_split_exact_text_overlap: {}};
const texts = {};
for (const split of ['train', 'dev', 'test']) {
  const name = `CMeEE_${split}.json`;
  const bytes = fs.readFileSync(path.join(root, 'data_raw', name));
  const rows = JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, ''));
  const counts = new Map();
  const s = {file: name, sha256: crypto.createHash('sha256').update(bytes).digest('hex'), records: rows.length, empty_text: 0, missing_entities: 0, empty_entities: 0, entities: 0, types: {}, invalid_spans: [], text_mismatches: [], duplicate_entities: 0, overlapping_entity_pairs: 0, nested_entity_pairs: 0, min_length: null, max_length: 0, total_characters: 0};
  rows.forEach((row, i) => {
    if (typeof row.text !== 'string') throw new Error(`${name}:${i}: text must be a string`);
    const chars = Array.from(row.text);
    counts.set(row.text, (counts.get(row.text) || 0) + 1);
    if (!chars.length) s.empty_text++;
    s.min_length = Math.min(s.min_length ?? chars.length, chars.length);
    s.max_length = Math.max(s.max_length, chars.length);
    s.total_characters += chars.length;
    if (!Array.isArray(row.entities)) { s.missing_entities++; return; }
    if (!row.entities.length) s.empty_entities++;
    const seen = new Set();
    for (const [j, e] of row.entities.entries()) {
      s.entities++;
      s.types[e.type] = (s.types[e.type] || 0) + 1;
      const key = JSON.stringify([e.start_idx, e.end_idx, e.type]);
      if (seen.has(key)) s.duplicate_entities++;
      seen.add(key);
      if (!Number.isInteger(e.start_idx) || !Number.isInteger(e.end_idx) || e.start_idx < 0 || e.end_idx < e.start_idx || e.end_idx >= chars.length) s.invalid_spans.push({row: i, entity: j});
      else if (chars.slice(e.start_idx, e.end_idx + 1).join('') !== e.entity) s.text_mismatches.push({row: i, entity: j});
    }
    for (let a = 0; a < row.entities.length; a++) for (let b = a + 1; b < row.entities.length; b++) {
      const x = row.entities[a], y = row.entities[b];
      if (Math.max(x.start_idx, y.start_idx) <= Math.min(x.end_idx, y.end_idx)) {
        s.overlapping_entity_pairs++;
        if ((x.start_idx <= y.start_idx && x.end_idx >= y.end_idx) || (y.start_idx <= x.start_idx && y.end_idx >= x.end_idx)) s.nested_entity_pairs++;
      }
    }
  });
  s.duplicate_text_extra_records = [...counts.values()].reduce((n, c) => n + c - 1, 0);
  s.mean_length = s.total_characters / s.records;
  texts[split] = new Set(counts.keys());
  report.files[split] = s;
}
for (const [a,b] of [['train','dev'],['train','test'],['dev','test']]) report.cross_split_exact_text_overlap[`${a}_${b}`] = [...texts[a]].filter(t => texts[b].has(t)).length;
fs.mkdirSync(path.join(root, 'docs'), {recursive: true});
fs.writeFileSync(path.join(root, 'docs', 'data_audit.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
