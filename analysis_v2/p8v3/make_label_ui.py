"""
p8v3.make_label_ui — build a single self-contained HTML file for BLIND human labelling.

Input: JSONL, one trajectory per line, already in display order (shuffled), e.g.
  {"audit_id": "A017", "config": "OpenHands/Qwen", "n_rounds_total": 41,
   "rounds": [{"round": 1, "calls": [{"tool": "execute_bash",
                                       "input": "cd /testbed && pytest -q",
                                       "output": "...observation text..."}]}, ...]}

Must NOT contain: parser outputs, sampling stratum, resolved label, trajectory_id
(trajectory_id lives only in the key file kept by the sampler).

Labelling is per ROUND. Each round has four events, each cycling
  blank -> 1 -> ? -> blank        (blank = 0 once the trajectory is marked Done)
Exports:
  labels_rounds.csv   audit_id,round,ran_tests,test_failed,build_or_collect_fail,exception
  labels.csv          audit_id,done,ran_tests_r5,...,exception_r15,first_test_round,notes
  where X_rK = 1 if any round<=K is 1; blank if none is 1 but some is "?"; else 0.

Usage:
  python -m p8v3.make_label_ui audit.jsonl label_ui.html
  python -m p8v3.make_label_ui audit.jsonl relabel_ui.html --ids A003,A044 --storage-key relabel1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVENTS = ["ran_tests", "test_failed", "build_or_collect_fail", "exception"]

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory Audit</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#deded8;--card:#fff;--acc:#2f5d8a;--yes:#2e7d4f;--unk:#b07a12;--code:#f3f3ef}
@media (prefers-color-scheme:dark){:root{--bg:#181816;--fg:#e8e8e3;--mut:#9a9a93;--line:#34342f;--card:#20201d;--acc:#7fb0e0;--yes:#5fc28a;--unk:#e0b04a;--code:#262622}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header b{font-size:15px}.sp{flex:1}
button,select{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
.prog{color:var(--mut);font-variant-numeric:tabular-nums}
main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:0;height:calc(100vh - 50px)}
#left{overflow:auto;padding:12px 16px}#right{overflow:auto;border-left:1px solid var(--line);padding:12px;background:var(--card)}
.round{border:1px solid var(--line);border-radius:8px;margin:0 0 12px;background:var(--card)}
.round h3{margin:0;padding:6px 10px;font-size:13px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
.call{padding:6px 10px;border-top:1px dashed var(--line)}.call:first-of-type{border-top:0}
.tool{font-size:12px;color:var(--mut)}
pre{margin:4px 0;padding:6px 8px;background:var(--code);border-radius:6px;white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;max-height:420px;overflow:auto}
pre.in{border-left:3px solid var(--acc)}
.tog{display:inline-flex;gap:4px;flex-wrap:wrap}
.t{font-size:11px;padding:2px 6px;border-radius:10px;border:1px solid var(--line);cursor:pointer;user-select:none;color:var(--mut)}
.t.v1{background:var(--yes);color:#fff;border-color:var(--yes)}.t.vq{background:var(--unk);color:#fff;border-color:var(--unk)}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--line);padding:3px 4px;text-align:left}
textarea{width:100%;min-height:110px;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
.done{color:var(--yes);font-weight:600}.muted{color:var(--mut);font-size:12px}
@media (max-width:820px){main{grid-template-columns:1fr;height:auto}#right{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header>
 <b id="aid"></b><span class="muted" id="meta"></span><span class="sp"></span>
 <span class="prog" id="prog"></span>
 <button id="prev">◀ Prev</button><select id="jump"></select><button id="next">Next ▶</button>
 <button id="nextTodo">Next unlabelled</button>
 <button class="pri" id="exp">Export CSVs</button>
 <label class="muted"><input type="file" id="imp" accept=".csv" style="width:170px"> import rounds CSV</label>
</header>
<main><div id="left"></div><aside id="right">
 <div><b>Summary (derived)</b></div>
 <table id="sum"></table>
 <p class="muted">Click event chips on each round: blank → 1 → ? → blank. “?” = cannot tell (e.g. truncated output).</p>
 <div><b>Notes</b></div><textarea id="notes" placeholder="boundary cases, truncation, unfamiliar framework…"></textarea>
 <p><label><input type="checkbox" id="done"> <b>Done</b> — blank chips on this trajectory mean 0</label></p>
 <p class="muted" id="saved"></p>
</aside></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const EV=__EVENTS__, KEY='p8audit:'+__KEY__, SHORT={ran_tests:'ran tests',test_failed:'test FAILED',build_or_collect_fail:'build/collect fail',exception:'exception'};
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){S={}}
let i=0;
const st=a=>(S[a]=S[a]||{r:{},notes:'',done:false});
function save(){try{localStorage.setItem(KEY,JSON.stringify(S));document.getElementById('saved').textContent='autosaved '+new Date().toLocaleTimeString()}catch(e){document.getElementById('saved').textContent='autosave unavailable — export often'}}
function el(t,c,txt){const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e}
function derive(a,K){const s=st(a),T=DATA.find(x=>x.audit_id===a);const o={};
 for(const ev of EV){let one=false,q=false;for(const r of T.rounds){if(r.round>K)continue;const v=(s.r[r.round]||{})[ev];if(v===1)one=true;if(v==='?')q=true}
  o[ev]=one?1:(q?'':(s.done?0:''))}
 return o}
function firstTest(a){const s=st(a),T=DATA.find(x=>x.audit_id===a);for(const r of T.rounds){if(r.round>15)continue;if((s.r[r.round]||{}).ran_tests===1)return r.round}return ''}
function renderSum(){const a=DATA[i].audit_id,t=document.getElementById('sum');t.innerHTML='';
 const h=el('tr');['event','≤5','≤15'].forEach(x=>h.appendChild(el('th',null,x)));t.appendChild(h);
 const d5=derive(a,5),d15=derive(a,15);
 for(const ev of EV){const tr=el('tr');tr.appendChild(el('td',null,SHORT[ev]));tr.appendChild(el('td',null,String(d5[ev])));tr.appendChild(el('td',null,String(d15[ev])));t.appendChild(tr)}
 const tr=el('tr');tr.appendChild(el('td',null,'first test round'));const td=el('td',null,String(firstTest(a)));td.colSpan=2;tr.appendChild(td);t.appendChild(tr)}
function render(){const T=DATA[i],s=st(T.audit_id);
 document.getElementById('aid').textContent=T.audit_id;
 document.getElementById('meta').textContent=`${T.config} · ${T.n_rounds_total} rounds total`+(s.done?' · ':'');
 if(s.done){const sp=el('span','done','✓ done');document.getElementById('meta').appendChild(sp)}
 const L=document.getElementById('left');const keep=L.scrollTop;L.innerHTML='';
 for(const r of T.rounds){const box=el('section','round');const h=el('h3');h.appendChild(el('span',null,'Round '+r.round));
  const tg=el('span','tog');const rv=(s.r[r.round]=s.r[r.round]||{});
  for(const ev of EV){const v=rv[ev];const c=el('span','t'+(v===1?' v1':v==='?'?' vq':''),SHORT[ev]+(v===1?' ✓':v==='?'?' ?':''));
   c.onclick=()=>{rv[ev]=rv[ev]===1?'?':rv[ev]==='?'?undefined:1;save();render()};tg.appendChild(c)}
  h.appendChild(tg);box.appendChild(h);
  for(const c of r.calls){const d=el('div','call');d.appendChild(el('div','tool',c.tool||'(tool)'));
   d.appendChild(el('pre','in',c.input==null?'':String(c.input)));d.appendChild(el('pre',null,c.output==null?'(no observation)':String(c.output)));box.appendChild(d)}
  L.appendChild(box)}
 L.scrollTop=keep;
 document.getElementById('notes').value=s.notes;document.getElementById('done').checked=s.done;
 document.getElementById('jump').value=String(i);
 const n=DATA.filter(x=>st(x.audit_id).done).length;document.getElementById('prog').textContent=`${i+1}/${DATA.length} · done ${n}`;
 renderSum()}
const J=document.getElementById('jump');DATA.forEach((x,k)=>{const o=el('option',null,x.audit_id);o.value=String(k);J.appendChild(o)});
J.onchange=()=>{i=+J.value;render();document.getElementById('left').scrollTop=0};
document.getElementById('prev').onclick=()=>{if(i>0){i--;render();document.getElementById('left').scrollTop=0}};
document.getElementById('next').onclick=()=>{if(i<DATA.length-1){i++;render();document.getElementById('left').scrollTop=0}};
document.getElementById('nextTodo').onclick=()=>{for(let k=1;k<=DATA.length;k++){const j=(i+k)%DATA.length;if(!st(DATA[j].audit_id).done){i=j;render();document.getElementById('left').scrollTop=0;return}}};
document.getElementById('notes').oninput=e=>{st(DATA[i].audit_id).notes=e.target.value;save()};
document.getElementById('done').onchange=e=>{st(DATA[i].audit_id).done=e.target.checked;save();render()};
const q=v=>{v=v==null?'':String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
function dl(name,txt){const b=new Blob([txt],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;document.body.appendChild(a);a.click();a.remove()}
document.getElementById('exp').onclick=()=>{
 let R=['audit_id,round,'+EV.join(',')];
 for(const T of DATA){const s=st(T.audit_id);for(const r of T.rounds){const rv=s.r[r.round]||{};R.push([T.audit_id,r.round,...EV.map(ev=>rv[ev]===1?1:rv[ev]==='?'?'?':(s.done?0:''))].map(q).join(','))}}
 const cols=['audit_id','done',...EV.map(e=>e+'_r5'),...EV.map(e=>e+'_r15'),'first_test_round','notes'];
 let Lb=[cols.join(',')];
 for(const T of DATA){const s=st(T.audit_id),d5=derive(T.audit_id,5),d15=derive(T.audit_id,15);
  Lb.push([T.audit_id,s.done?1:0,...EV.map(e=>d5[e]),...EV.map(e=>d15[e]),firstTest(T.audit_id),s.notes].map(q).join(','))}
 dl('labels_rounds'+__SUFFIX__+'.csv',R.join('\n')+'\n');setTimeout(()=>dl('labels'+__SUFFIX__+'.csv',Lb.join('\n')+'\n'),300)};
document.getElementById('imp').onchange=async e=>{const f=e.target.files[0];if(!f)return;const lines=(await f.text()).trim().split(/\r?\n/);const hd=lines[0].split(',');
 for(const ln of lines.slice(1)){const c=ln.split(',');const a=c[0],r=+c[1];const s=st(a);const rv=(s.r[r]=s.r[r]||{});
  hd.slice(2).forEach((ev,k)=>{const v=c[k+2];rv[ev]=v==='1'?1:v==='?'?'?':undefined})}
 save();render();alert('Imported. Note: Done flags and notes are not in the rounds CSV — re-check them.')};
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowRight')document.getElementById('next').click();if(e.key==='ArrowLeft')document.getElementById('prev').click()});
render();
</script></body></html>
"""

FORBIDDEN_KEYS = {"trajectory_id", "resolved", "label", "stratum", "parser", "parser_v2",
                  "feature_value", "screen", "issue_id", "repo"}


def build(jsonl: Path, out: Path, ids: list[str] | None = None,
          storage_key: str = "main", suffix: str = "") -> int:
    recs = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ids:
        keep = set(ids)
        recs = [r for r in recs if r["audit_id"] in keep]
    for r in recs:
        bad = FORBIDDEN_KEYS & set(r)
        if bad:
            raise ValueError(f"{r.get('audit_id')}: blind-labelling input must not contain {bad}")
        for k in ("audit_id", "config", "n_rounds_total", "rounds"):
            if k not in r:
                raise ValueError(f"record missing {k}")
    data = json.dumps(recs, ensure_ascii=False).replace("</", "<\\/")
    html = (HTML.replace("__DATA__", data)
                .replace("__EVENTS__", json.dumps(EVENTS))
                .replace("__KEY__", json.dumps(storage_key))
                .replace("__SUFFIX__", json.dumps(suffix)))
    out.write_text(html, encoding="utf-8")
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ids", default=None, help="comma-separated audit_ids (relabel subset)")
    ap.add_argument("--storage-key", default="main")
    ap.add_argument("--suffix", default="", help="appended to exported file names, e.g. _relabel")
    a = ap.parse_args()
    n = build(a.jsonl, a.out, a.ids.split(",") if a.ids else None, a.storage_key, a.suffix)
    print(f"wrote {a.out} with {n} trajectories")


if __name__ == "__main__":
    main()
