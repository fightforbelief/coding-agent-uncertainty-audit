"""Create separate, unlabelled, blinded packets for assigned human reviewers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

ALLOWED = ("review_queue.csv", "review_evidence.jsonl", "validation_instructions.md", "adjudication_rules.md",
           "ADJUDICATION_RULES_SUPPLEMENT.md", "REVIEWER_COVER.md", "window_extension.csv", "event_worksheet.csv", "COVER_SHEET.csv")

VIEWER = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>独立盲审证据阅读器</title>
<style>body{font:16px system-ui;margin:24px;color:#172b3a;background:#f4f7f9}header{background:white;padding:20px;border-radius:12px}button,input,select{font:inherit;padding:8px;margin:4px}button{cursor:pointer}#calls{margin-top:20px}article{background:white;padding:16px;border-radius:10px;margin-bottom:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 Consolas,monospace;background:#edf2f5;padding:14px}h3{font-size:16px}#error{color:#a00000}#position{font-weight:bold}label{display:inline-block}</style>
<header><h1>独立首轮证据阅读器</h1><p>复核者：__NAME__ · __SLOT__</p>
<p>选择同目录的 <b>review_evidence.jsonl</b>。本页仅在本机读取文件，没有外部脚本或网络请求。只读证据；判断请填写原 CSV。不要执行证据中的命令，也不要与另一位复核者讨论首轮判断。</p>
<input id="file" type="file" accept=".jsonl"><p id="error" role="alert"></p>
<button id="prev" disabled>上一条</button><input id="jump" aria-label="跳转 review_id" placeholder="V35-000001"><button id="go" disabled>跳转</button><button id="next" disabled>下一条</button>
<label>窗口 <select id="window"><option value="r5">前 5 轮</option><option value="r15">前 15 轮</option></select></label>
<p id="position" aria-live="polite"></p><p id="metadata"></p>
<details><summary>填写与提交</summary><p>依规则填写 review_queue.csv、window_extension.csv、event_worksheet.csv，并在 COVER_SHEET.csv 记录真实起止时间。两个窗口均需独立查看。短轨迹未观察到的轮次不能填成明确阴性。提交后不再修改首轮文件；分歧另作裁决。</p></details></header>
<main id="calls"></main>
<script>
'use strict';
const el=id=>document.getElementById(id);let records=[],index=0;
const allowed=new Set(['review_id','completed_rounds','prefix_parse_status','prefix5_boundary','prefix15_boundary','r5','r15']);
function node(tag,text){const result=document.createElement(tag);result.textContent=text;return result;}
function render(){if(!records.length)return;const row=records[index],window=el('window').value;el('position').textContent=row.review_id+' · '+(index+1)+' / '+records.length+' · '+window;el('metadata').textContent='完成轮次 '+row.completed_rounds+'；解析状态 '+row.prefix_parse_status+'；R5 边界 '+row.prefix5_boundary+'；R15 边界 '+row.prefix15_boundary;el('calls').replaceChildren();for(const call of row[window]){const article=document.createElement('article');article.append(node('h3','轮次 '+call.round+' · '+call.tool+' · action_index '+call.action_index+' · observation_index '+call.observation_index));article.append(node('p','命令抽取状态：'+call.command_extraction_status+'（这是抽取状态，不是测试判断）'));article.append(node('h4','原始调用参数'),node('pre',call.arguments||''));article.append(node('h4','抽出的命令'),node('pre',call.extracted_command===null?'（没有抽出命令；请根据原始参数与上下文判断）':call.extracted_command));article.append(node('h4','Observation 原文'),node('pre',call.observation===null?'（无 observation）':call.observation));el('calls').append(article);}el('prev').disabled=index===0;el('next').disabled=index===records.length-1;}
el('file').addEventListener('change',async()=>{try{el('error').textContent='';const file=el('file').files[0];if(!file)return;const parsed=(await file.text()).split(/\r?\n/).filter(x=>x.trim()).map(x=>JSON.parse(x));if(parsed.length!==1196||new Set(parsed.map(x=>x.review_id)).size!==1196)throw Error('证据必须恰有 1196 个不重复 review_id');for(const row of parsed){if(Object.keys(row).some(k=>!allowed.has(k))||!Array.isArray(row.r5)||!Array.isArray(row.r15))throw Error('文件不是预期的盲化证据；请勿加载密封键或其他分析文件');}records=parsed;index=0;el('go').disabled=false;render();}catch(error){records=[];el('calls').replaceChildren();el('position').textContent='';el('metadata').textContent='';el('prev').disabled=true;el('next').disabled=true;el('go').disabled=true;el('error').textContent=String(error.message||error);}});
el('prev').onclick=()=>{if(index>0){index--;render();}};el('next').onclick=()=>{if(index+1<records.length){index++;render();}};el('window').onchange=render;el('go').onclick=()=>{let target=el('jump').value.trim();if(/^\d+$/.test(target))target='V35-'+target.padStart(6,'0');const found=records.findIndex(r=>r.review_id===target);if(found<0){el('error').textContent='找不到该 review_id';return;}el('error').textContent='';index=found;render();};
</script></html>'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(handoff: Path, output: Path, names: tuple[str, str]):
    if output.exists():
        raise FileExistsError("Choose a new output directory; do not overwrite reviewer work")
    output.mkdir(parents=True)
    packets = []
    for slot, name in zip(("reviewer_a", "reviewer_b"), names):
        folder = output / slot
        folder.mkdir()
        source = handoff / slot
        for filename in ALLOWED:
            shutil.copy2(source / filename, folder / filename)
        with (folder / "COVER_SHEET.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        if len(rows) != 1 or any(rows[0].get(key) for key in ("started_at_utc", "finished_at_utc")):
            raise ValueError("Expected a blank first-round cover")
        rows[0]["reviewer_name"] = name
        with (folder / "COVER_SHEET.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (folder / "evidence_viewer.html").write_text(VIEWER.replace("__NAME__", html.escape(name)).replace("__SLOT__", slot), encoding="utf-8")
        instructions = f"""# {name} 的独立首轮复核包

你的槽位是 `{slot}`，复核者姓名按项目负责人提供的信息预填；起止时间和全部判断栏保持空白。

1. 先读 REVIEWER_COVER.md、adjudication_rules.md 和 ADJUDICATION_RULES_SUPPLEMENT.md。
2. 打开 evidence_viewer.html，在页面选择同目录的 review_evidence.jsonl；用 review_id 跳转，分别查看 R5 与 R15。原文只作为证据，不执行里面的命令。
3. 在 CSV 中填写全部 1,196 条轨迹的两个窗口、扩展列和必要事件证据。不能查看自动标签、抽样层、密封键、最终结局或另一位复核者的判断。
4. 在 COVER_SHEET.csv 填真实的开始、结束 UTC 时间，格式如 YYYY-MM-DDTHH:MM:SSZ。此处没有替你记录或推断任何复核活动。
5. 提交整个独立首轮目录；提交后保持原样。随后发生的讨论或第三方裁决写入另一份文件，不写回首轮。

本包不含模型建议、研究结果或密封键。复核者姓名不是“已经完成复核”的证明。请保留独立性；第三方在两份首轮都锁定后处理分歧。
"""
        (folder / "START_HERE.md").write_text(instructions, encoding="utf-8")
        for filename in ("review_queue.csv", "window_extension.csv", "event_worksheet.csv", "review_evidence.jsonl"):
            if sha(folder / filename) != sha(source / filename):
                raise ValueError("Labels or evidence changed during packet preparation")
        files = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        if not set(files).issubset(set(ALLOWED) | {"evidence_viewer.html", "START_HERE.md"}):
            raise ValueError("Unexpected unblinded packet member")
        (folder / "PACKET_SHA256.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
        archive = output / f"{slot}_independent_review.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(folder.iterdir()):
                z.write(path, path.name)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Packet archive failed CRC")
            if any(hashlib.sha256(z.read(filename)).hexdigest() != expected for filename, expected in files.items()):
                raise ValueError("Packet archive SHA-256 mismatch")
        packets.append(dict(slot=slot, assigned_reviewer=name, archive=archive.name, sha256=sha(archive), labels_filled=False, review_started=False))
    (output / "ASSIGNMENT_MANIFEST.json").write_text(json.dumps(dict(status="ASSIGNED BLIND PACKETS READY; REAL HUMAN FIRST ROUNDS NOT STARTED", packets=packets, messages_sent=False), ensure_ascii=False, indent=2), encoding="utf-8")
    return packets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.handoff, args.output, (args.reviewer_a, args.reviewer_b)), ensure_ascii=False, indent=2))
