r"""ScriptSense Local Demonstration Application

Paste any text and get AI-writing detection + plagiarism analysis.

Run:
    .\.venv\Scripts\python.exe demo_app.py --web
"""
from __future__ import annotations

import argparse
import http.server
import json
import math
import re
import socketserver
import sys
import webbrowser
from pathlib import Path

# Add ml to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "ml"))

from scriptsense_ml.data.normalization import normalize_for_model, word_count
from scriptsense_ml.inference.chunking import segment_blocks, eligible_spans
from scriptsense_ml.similarity.fingerprints import tokenize, align_exact
from scriptsense_ml.similarity.lexical_search import BM25Index

# ── Reference corpus (papers the similarity engine checks against) ──────────
REFERENCE_CORPUS = {
    "paper:001": (
        "Machine learning models often suffer from shortcut learning, where classifiers rely on spurious "
        "correlations rather than robust semantic features. This phenomenon is especially prevalent in text "
        "classification when training datasets exhibit distribution shifts between synthetic generation prompts."
    ),
    "paper:002": (
        "Coastal wetland ecosystems provide vital storm surge buffering and carbon sequestration services. "
        "Longitudinal monitoring across estuaries reveals that sediment accretion rates rarely keep pace with "
        "accelerated relative sea-level rise without inland migration corridors."
    ),
    "paper:003": (
        "Distributed consensus protocols in permissionless networks require economic sybil resistance mechanisms. "
        "Proof-of-work ties block creation voting weight to computational energy expenditure, whereas proof-of-stake "
        "substitutes thermodynamic expenditure with bonded collateral deposits."
    ),
}

corpus_index = BM25Index()
for doc_id, text in REFERENCE_CORPUS.items():
    corpus_index.add(doc_id, text)


# ── Check if the real trained model exists ──────────────────────────────────
MODEL_DIR = None
_analyzer = None

def _find_trained_model():
    """Look for a trained+calibrated model in the standard artifact locations."""
    candidates = [
        Path.home() / "scriptsense_data" / "artifacts",
        Path.home() / "scriptsense_notebook" / "data" / "artifacts",
    ]
    env_art = Path(
        __import__("os").environ.get("SCRIPTSENSE_ARTIFACT_DIR", "")
    )
    if env_art.name:
        candidates.insert(0, env_art)

    for art_dir in candidates:
        runs_dir = art_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            best = run_dir / "best"
            if (best / "heads.pt").exists() and (best / "calibration.json").exists():
                return best
    return None


def _get_analyzer():
    global _analyzer, MODEL_DIR
    if _analyzer is not None:
        return _analyzer
    MODEL_DIR = _find_trained_model()
    if MODEL_DIR is None:
        return None
    from scriptsense_ml.inference.service import AIWritingAnalyzer
    _analyzer = AIWritingAnalyzer(MODEL_DIR)
    print(f"[*] Loaded trained model from: {MODEL_DIR}")
    return _analyzer


# ── Linguistic signal heuristics (fallback when no trained model) ───────────
def compute_linguistic_signals(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"burstiness": 0.0, "ttr": 0.0, "avg_sentence_len": 0.0, "sentence_len_std": 0.0}
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences] if sentences else [len(words)]
    avg = sum(lens) / len(lens) if lens else 0
    var = sum((l - avg) ** 2 for l in lens) / len(lens) if lens else 0
    std = math.sqrt(var)
    return {
        "burstiness": round(std / avg if avg > 0 else 0.0, 3),
        "ttr": round(len(set(words)) / len(words), 3),
        "avg_sentence_len": round(avg, 1),
        "sentence_len_std": round(std, 1),
    }


# ── Core analysis (uses real model if available, otherwise heuristic) ───────
def analyze_submission(text: str) -> dict:
    blocks = segment_blocks(text)
    spans = eligible_spans(blocks)
    total_words = word_count(text)
    eligible_words = sum(word_count(text[s:e]) for s, e in spans)

    excluded_summary = {}
    for b in blocks:
        if not b.eligible:
            excluded_summary[b.reason] = excluded_summary.get(b.reason, 0) + word_count(text[b.start:b.end])

    # Similarity / Plagiarism (always real — Winnowing algorithm)
    sub_tokens = tokenize(text)
    similarity_matches = []
    total_matched_words = 0
    for doc_id, ref_text in REFERENCE_CORPUS.items():
        for m in align_exact(sub_tokens, tokenize(ref_text), n=4, min_words=6):
            total_matched_words += m.n_words
            similarity_matches.append({
                "source_id": doc_id,
                "n_words": m.n_words,
                "sub_start": m.sub_start, "sub_end": m.sub_end,
                "sub_text": text[m.sub_start:m.sub_end],
                "src_text": ref_text[m.src_start:m.src_end],
            })
    similarity_score = round(min(100.0, total_matched_words / max(total_words, 1) * 100), 1)

    signals = compute_linguistic_signals(text)

    # ── Try real model inference ────────────────────────────────────────
    analyzer = _get_analyzer()
    using_real_model = analyzer is not None

    if using_real_model:
        result = analyzer.analyze(text, page_starts=[0], document_id="demo")
        calibrated_score = result.get("ai_signal_score", 0.0)
        is_flagged = result.get("status") == "flagged"
        predicted_class = result.get("document_class", "HUMAN_ONLY")
        probs = result.get("class_probabilities", {})
        class_probs = {
            "human_only": probs.get("human_only", 0.0),
            "ai_generated": probs.get("ai_generated", 0.0),
            "ai_paraphrased": probs.get("ai_paraphrased_or_refined", 0.0),
            "mixed_assisted": probs.get("mixed_or_ai_assisted", 0.0),
        }
        operating_threshold = result.get("threshold", 0.65)
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "type": "flagged" if seg.get("flagged") else "clean",
                "reason": seg.get("predicted_class", ""),
                "text": text[seg["start"]:seg["end"]],
                "score": round(seg.get("ai_signal", 0.0), 3),
            })
        # Also add excluded blocks
        for b in blocks:
            if not b.eligible:
                segments.insert(0, {
                    "type": "excluded", "reason": b.reason,
                    "text": text[b.start:b.end], "score": 0.0,
                })
    else:
        # ── Heuristic fallback ──────────────────────────────────────────
        b_val = signals["burstiness"]
        ai_clues = sum(
            0.15 for p in [
                r"\bfurthermore\b", r"\bin conclusion\b", r"\bit is important to\b",
                r"\bcrucial role\b", r"\bholistic\b", r"\bunprecedented\b", r"\bfostering\b",
            ] if re.search(p, text, re.I)
        )
        raw = max(0.02, min(0.98, (0.75 - b_val * 0.8) + ai_clues * 0.6))
        if total_matched_words > 15:
            raw = 0.08
        calibrated_score = round(1 / (1 + math.exp(-math.log(raw / (1 - raw + 1e-9)) / 1.4)), 3)
        operating_threshold = 0.65
        is_flagged = calibrated_score >= operating_threshold

        if calibrated_score >= operating_threshold:
            c_ai = round(calibrated_score * 0.75, 3)
            c_para = round(calibrated_score * 0.20, 3)
            c_mix = round(calibrated_score * 0.05, 3)
            c_hum = round(1 - c_ai - c_para - c_mix, 3)
            predicted_class = "AI_GENERATED"
        elif calibrated_score > 0.35:
            c_hum = round(1 - calibrated_score, 3)
            c_para = round(calibrated_score * 0.55, 3)
            c_mix = round(calibrated_score * 0.30, 3)
            c_ai = round(calibrated_score * 0.15, 3)
            predicted_class = "AI_PARAPHRASED_OR_REFINED"
        else:
            c_hum = round(1 - calibrated_score, 3)
            c_mix = round(calibrated_score * 0.4, 3)
            c_para = round(calibrated_score * 0.4, 3)
            c_ai = round(calibrated_score * 0.2, 3)
            predicted_class = "HUMAN_ONLY"
        class_probs = {"human_only": c_hum, "ai_generated": c_ai, "ai_paraphrased": c_para, "mixed_assisted": c_mix}

        segments = []
        for b_item in blocks:
            bt = text[b_item.start:b_item.end]
            if not b_item.eligible:
                segments.append({"type": "excluded", "reason": b_item.reason, "text": bt, "score": 0.0})
            else:
                ss = compute_linguistic_signals(bt)
                sa = max(0.01, min(0.99, (0.75 - ss["burstiness"] * 0.8) + ai_clues * 0.5))
                segments.append({
                    "type": "flagged" if sa >= operating_threshold else "clean",
                    "reason": "eligible", "text": bt, "score": round(sa, 3),
                })

    return {
        "total_words": total_words,
        "eligible_words": eligible_words,
        "excluded_summary": excluded_summary,
        "similarity_score_pct": similarity_score,
        "similarity_matches": similarity_matches,
        "calibrated_ai_score": calibrated_score,
        "operating_threshold": operating_threshold,
        "is_flagged": is_flagged,
        "predicted_class": predicted_class,
        "class_probabilities": class_probs,
        "linguistic_signals": signals,
        "segments": segments,
        "using_real_model": using_real_model,
        "model_architecture": "DeBERTa-v3 Multi-Task (4-way provenance + binary head)",
        "similarity_engine": "Winnowing Fingerprinting (Schleimer et al., 2003) + BM25",
    }


# ── HTML ────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScriptSense ML Engine</title>
<style>
  :root { --bg:#090d16; --card:#111827; --bdr:#1f293d; --cy:#06b6d4; --grn:#10b981; --red:#ef4444; --amb:#f59e0b; --txt:#f3f4f6; --mut:#9ca3af; }
  *{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  body{background:var(--bg);color:var(--txt);padding:24px;min-height:100vh}
  .container{max-width:1200px;margin:0 auto}
  header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--bdr);padding-bottom:16px;margin-bottom:24px}
  .logo{font-size:24px;font-weight:800;color:#fff;display:flex;align-items:center;gap:8px} .logo span{color:var(--cy)}
  .badge{background:rgba(6,182,212,.15);border:1px solid var(--cy);color:var(--cy);padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:24px} @media(max-width:850px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--bdr);border-radius:12px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,.5)}
  h2{font-size:18px;margin-bottom:12px;color:#fff;display:flex;align-items:center;justify-content:space-between}
  textarea{width:100%;height:320px;background:#0b1120;border:1px solid #334155;border-radius:8px;color:#fff;padding:12px;font-size:14px;line-height:1.5;resize:vertical}
  textarea:focus{outline:none;border-color:var(--cy);box-shadow:0 0 10px rgba(6,182,212,.25)}
  .btn-analyze{width:100%;background:linear-gradient(135deg,#06b6d4,#2563eb);border:none;color:#fff;font-weight:700;padding:14px;border-radius:8px;margin-top:12px;font-size:15px;cursor:pointer}
  .btn-analyze:hover{opacity:.9}
  .metric-tile{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#0f172a;border-radius:8px;margin-bottom:8px;border:1px solid #1e293b}
  .metric-label{font-size:13px;color:var(--mut)} .metric-value{font-size:16px;font-weight:700}
  .score-gauge{text-align:center;padding:20px;background:#0f172a;border-radius:12px;margin-bottom:16px;border:1px solid #1e293b}
  .score-big{font-size:48px;font-weight:900;margin:8px 0}
  .score-safe{color:var(--grn)} .score-warn{color:var(--amb)} .score-danger{color:var(--red)}
  .highlight-box{background:#0b1120;border:1px solid #1e293b;border-radius:8px;padding:14px;max-height:300px;overflow-y:auto;font-size:14px;line-height:1.6}
  .hl-flagged{background:rgba(239,68,68,.25);border-bottom:2px solid var(--red)}
  .hl-plagiarized{background:rgba(245,158,11,.25);border-bottom:2px solid var(--amb)}
  .hl-excluded{background:rgba(156,163,175,.15);color:#9ca3af;font-style:italic}
  .model-banner{grid-column:span 2;text-align:center;padding:10px;border-radius:8px;font-size:13px;font-weight:600}
  @media(max-width:850px){.model-banner{grid-column:span 1}}
  .banner-real{background:rgba(16,185,129,.15);border:1px solid var(--grn);color:var(--grn)}
  .banner-heuristic{background:rgba(245,158,11,.15);border:1px solid var(--amb);color:var(--amb)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">Script<span>Sense</span></div>
    <div class="badge">ML Pipeline (Local GPU)</div>
  </header>
  <div class="grid">
    <div class="card">
      <h2>Paste Text to Analyze</h2>
      <textarea id="inputText" placeholder="Paste any text here -- essay, assignment, article -- and click Analyze..."></textarea>
      <button class="btn-analyze" onclick="analyze()">Analyze</button>
    </div>
    <div class="card">
      <h2>Results <span id="verdictBadge" class="badge">Awaiting Input</span></h2>
      <div class="score-gauge">
        <div style="font-size:13px;color:var(--mut)">AI-Writing Signal</div>
        <div id="aiScoreVal" class="score-big score-safe">--</div>
        <div id="aiVerdict" style="font-size:14px;font-weight:600">Paste text and click Analyze</div>
      </div>
      <div class="metric-tile"><div class="metric-label">Similarity (Winnowing)</div><div id="simScoreVal" class="metric-value" style="color:var(--amb)">--</div></div>
      <div class="metric-tile"><div class="metric-label">Predicted Class</div><div id="predClassVal" class="metric-value">--</div></div>
      <div class="metric-tile"><div class="metric-label">Burstiness</div><div id="burstVal" class="metric-value">--</div></div>
      <div class="metric-tile"><div class="metric-label">Words (eligible / total)</div><div id="wordsVal" class="metric-value">--</div></div>
      <h2 style="margin-top:16px">Passage Highlights</h2>
      <div id="highlightView" class="highlight-box">Flagged and matched passages will appear here.</div>
    </div>
    <div id="modelBanner" class="model-banner banner-heuristic">Loading...</div>
  </div>
</div>
<script>
async function analyze(){
  const text=document.getElementById('inputText').value;
  if(!text.trim())return;
  document.querySelector('.btn-analyze').textContent='Analyzing...';
  const res=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}).then(r=>r.json());
  document.querySelector('.btn-analyze').textContent='Analyze';
  const pct=Math.round(res.calibrated_ai_score*100);
  const el=document.getElementById('aiScoreVal');
  el.innerText=pct+'%';
  el.className='score-big '+(res.is_flagged?'score-danger':(pct>35?'score-warn':'score-safe'));
  document.getElementById('aiVerdict').innerText=res.is_flagged?'FLAGGED':'CLEAN';
  document.getElementById('verdictBadge').innerText=res.predicted_class;
  document.getElementById('verdictBadge').style.borderColor=res.is_flagged?'#ef4444':'#10b981';
  document.getElementById('verdictBadge').style.color=res.is_flagged?'#ef4444':'#10b981';
  document.getElementById('simScoreVal').innerText=res.similarity_score_pct+'%';
  document.getElementById('predClassVal').innerText=res.predicted_class;
  document.getElementById('burstVal').innerText=res.linguistic_signals.burstiness;
  document.getElementById('wordsVal').innerText=res.eligible_words+' / '+res.total_words;
  let h='';
  res.segments.forEach(s=>{let c='';if(s.type==='excluded')c='hl-excluded';else if(s.type==='flagged')c='hl-flagged';h+=`<span class="${c}" title="${s.reason}">${s.text} </span>`;});
  if(res.similarity_matches.length>0){h+='<br><br><strong>Matched Passages:</strong><ul>';res.similarity_matches.forEach(m=>{h+=`<li><span class="hl-plagiarized">"${m.sub_text}"</span> (${m.n_words} words) matched [${m.source_id}]</li>`;});h+='</ul>';}
  document.getElementById('highlightView').innerHTML=h;
  const ban=document.getElementById('modelBanner');
  if(res.using_real_model){ban.className='model-banner banner-real';ban.innerText='Using trained DeBERTa-v3 model';}
  else{ban.className='model-banner banner-heuristic';ban.innerText='Using heuristic scoring (train the model on Colab for real inference)';}
}
fetch('/api/status').then(r=>r.json()).then(d=>{
  const ban=document.getElementById('modelBanner');
  if(d.model_loaded){ban.className='model-banner banner-real';ban.innerText='Trained DeBERTa-v3 model loaded from: '+d.model_path;}
  else{ban.className='model-banner banner-heuristic';ban.innerText='No trained model found -- using heuristic scoring. Train the model on Google Colab for real inference.';}
});
</script>
</body>
</html>
"""


# ── HTTP Server ─────────────────────────────────────────────────────────────
class ScriptSenseServer(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/api/status":
            analyzer = _get_analyzer()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "model_loaded": analyzer is not None,
                "model_path": str(MODEL_DIR) if MODEL_DIR else None,
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/analyze":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            result = analyze_submission(data.get("text", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_web_demo(port: int = 8000):
    # Try to load the real model at startup
    analyzer = _get_analyzer()
    if analyzer:
        print(f"[*] Real DeBERTa model loaded from: {MODEL_DIR}")
    else:
        print("[*] No trained model found. Using heuristic scoring.")
        print("[*] Train the model on Google Colab to enable real DeBERTa inference.")

    with socketserver.TCPServer(("", port), ScriptSenseServer) as httpd:
        url = f"http://127.0.0.1:{port}"
        print(f"\n=======================================================")
        print(f"[*] ScriptSense running at: {url}")
        print(f"[*] Press Ctrl+C to stop.")
        print(f"=======================================================\n")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScriptSense")
    parser.add_argument("--web", action="store_true", help="Launch web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()
    if args.web:
        run_web_demo(args.port)
    else:
        print("Run with --web to launch the browser UI")
