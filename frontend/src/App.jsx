"use client";
// src/App.jsx
import React, { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const IOU_FIXED = 0.6;
const IMGSZ_FIXED = 1024;

const LS_KEY = "bidai_history_v1";
const MAX_ITEMS = 12;        // максимум записей в истории
const MAX_JSON_KB = 4500;    // лимит на общий размер истории (~4.5MB)
const BIG_DATAURL = 2_000_000; // ~2MB — слишком большой dataURL для истории

// компактная превьюшка (уменьшаем до нужной ширины, jpegQuality 0..1)
async function makeThumb(dataUrl, maxW = 640, jpegQ = 0.8) {
  if (!dataUrl) return null;
  try {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = dataUrl;
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });

    const scale = Math.min(1, maxW / img.width);
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, w, h);
    return c.toDataURL("image/jpeg", jpegQ);
  } catch {
    return null;
  }
}

function useLocalHistory() {
  const [items, setItems] = useState(() => {
    try {
      const raw = localStorage.getItem(LS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });

  // безопасная запись с ограничением размера
  useEffect(() => {
    try {
      const json = JSON.stringify(items);
      const kb = new Blob([json]).size / 1024;
      if (kb > MAX_JSON_KB) {
        const trimmed = items.slice(0, Math.max(3, Math.floor(items.length * 0.7)));
        localStorage.setItem(LS_KEY, JSON.stringify(trimmed));
        setItems(trimmed);
      } else {
        localStorage.setItem(LS_KEY, json);
      }
    } catch (e) {
      if (e?.name === "QuotaExceededError") {
        const trimmed = items.slice(0, Math.max(3, Math.floor(items.length * 0.7)));
        try {
          localStorage.setItem(LS_KEY, JSON.stringify(trimmed));
          setItems(trimmed);
        } catch { /* игнор */ }
      }
    }
  }, [items]);

  const add = (it) => setItems(prev =>
    [{ id: crypto.randomUUID(), ...it }, ...prev].slice(0, MAX_ITEMS)
  );
  const remove = (id) => setItems(prev => prev.filter(x => x.id !== id));
  const clear = () => setItems([]);
  return { items, add, remove, clear };
}

export default function App() {
  const [tab, setTab] = useState("analyze"); // "analyze" | "history"

  // analyze state
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resultImg, setResultImg] = useState(null);
  const [count, setCount] = useState(null);
  const [kgHa, setKgHa] = useState(null);
  const [conf, setConf] = useState(0.35);
  const [frameCm, setFrameCm] = useState(50);
  const [tkw, setTkw] = useState(40);

  const { items: history, add: addHistory, remove: removeHistory, clear: clearHistory } = useLocalHistory();

  // стили (белое + золото)
  useEffect(() => {
    if (typeof document === "undefined") return;
    const s = document.createElement("style");
    s.innerHTML = `
      body{background:#fffdf6;margin:0;font-family:Inter,system-ui,Arial;color:#14120b}
      .wrap{max-width:980px;margin:40px auto 100px;padding:0 20px}
      .badge{display:inline-block;padding:6px 14px;background:linear-gradient(90deg,#d6b341,#f9e98c);
         border-radius:999px;font-weight:700;color:#111;margin-bottom:8px}
      .title{font-size:32px;font-weight:900;margin:0 0 18px}
      .tabs{display:flex;gap:8px;margin:18px 0}
      .tab{padding:10px 14px;border-radius:12px;border:1px solid #eee;background:#fff;cursor:pointer;font-weight:700;color:#6b6b6b;transition:all .14s;}
      .tab:hover{background:#f9e98c;color:#111;}
      .tab.active{background:linear-gradient(90deg,#d6b341,#f5d866);color:#14120b;}
      .card{background:#fff;border:1px solid #eee;border-radius:20px;padding:22px 20px;box-shadow:0 10px 28px rgba(0,0,0,.06);margin-bottom:24px}
      .sub{font-size:20px;font-weight:800;margin:0 0 14px}
      .drop{border:2px dashed #e8c440;background:#fffef7;border-radius:16px;padding:28px;text-align:center;color:#555;transition:.2s}
      .drop:hover,.drop.hover{background:#fff9df;box-shadow:inset 0 0 0 2px rgba(232,196,64,.25)}
      .drop input{display:none}
      .controls{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
      .controls label{font-weight:700;color:#333;display:flex;flex-direction:column;gap:6px}
      .controls input{background:#fff;border:1px solid #e7e7e7;border-radius:12px;padding:12px 14px;font-size:15px;color:#111}
      .btnPrimary{background:linear-gradient(90deg,#d6b341,#f5d866);color:#14120b;font-weight:900;border:none;border-radius:14px;padding:14px 18px;margin-top:16px;width:100%;font-size:16px;box-shadow:0 10px 22px rgba(214,179,65,.28);cursor:pointer;transition:all .14s;}
      .btnPrimary:disabled{filter:grayscale(.2) brightness(.96) opacity(.7);cursor:not-allowed}
      .btnPrimary:hover{filter:brightness(1.03)}
      .img{display:block;width:100%;height:560px;object-fit:contain;border-radius:16px;border:1px solid #eee}
      .metrics{display:flex;gap:16px;margin-top:24px}
      .metric{flex:1;background:#fffef6;border:1px solid #f0e6b2;border-radius:16px;text-align:center;padding:16px 14px}
      .metric .cap{color:#6b6b6b;font-size:13px;font-weight:700;letter-spacing:.2px}
      .metric .val{margin-top:6px;font-size:36px;font-weight:900;color:#cfae2b}
      .err{background:#fff2cc;border:1px solid #f0d47a;color:#7a4f00;border-radius:12px;padding:10px 12px}
      .row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
      .thumb{width:100%;height:140px;object-fit:cover;border-radius:12px;border:1px solid #eee;background:#fafafa}
      .hitem{display:flex;gap:14px;border:1px solid #eee;border-radius:14px;padding:12px;background:#fff}
      .hmeta{display:flex;gap:14px;flex-wrap:wrap;color:#555;font-size:13px}
      .hbtns{display:flex;gap:8px;margin-top:8px}
      .btn{padding:8px 12px;border:1px solid #e7e7e7;border-radius:10px;background:#fff;cursor:pointer;font-weight:700;color:#14120b;transition:all .14s;}
      .btn:hover{background:#f9e98c;color:#111;filter:brightness(1.03);}
      .btnGold{background:linear-gradient(90deg,#d6b341,#f5d866);border:none;color:#14120b;font-weight:800;box-shadow:0 8px 22px rgba(214,179,65,.12);transition:all .14s;}
      @media(max-width:900px){.controls{grid-template-columns:1fr 1fr}.row{grid-template-columns:1fr}}
      @media(max-width:600px){.controls{grid-template-columns:1fr}}
    `;
    document.head.appendChild(s);
    return () => document.head.removeChild(s);
  }, []);

  // Drag & Drop — один ref и один эффект (никаких хуков внутри условий!)
  const dropRef = useRef(null);
  useEffect(() => {
    const el = dropRef.current; if (!el) return;
    const preventDoc = (e) => { e.preventDefault(); };
    window.addEventListener('dragover', preventDoc);
    window.addEventListener('drop', preventDoc);

    const enter = (e) => { e.preventDefault(); e.stopPropagation(); el.classList.add('hover'); };
    const over  = (e) => { e.preventDefault(); e.stopPropagation(); };
    const leave = (e) => { e.preventDefault(); e.stopPropagation(); el.classList.remove('hover'); };
    const drop  = (e) => {
      e.preventDefault(); e.stopPropagation(); el.classList.remove('hover');
      const f = e.dataTransfer?.files?.[0];
      if (f) { setFile(f); setPreview(URL.createObjectURL(f));
        setResultImg(null); setCount(null); setKgHa(null); setError(""); }
    };
    el.addEventListener('dragenter', enter);
    el.addEventListener('dragover', over);
    el.addEventListener('dragleave', leave);
    el.addEventListener('drop', drop);

    return () => {
      window.removeEventListener('dragover', preventDoc);
      window.removeEventListener('drop', preventDoc);
      el.removeEventListener('dragenter', enter);
      el.removeEventListener('dragover', over);
      el.removeEventListener('dragleave', leave);
      el.removeEventListener('drop', drop);
    };
  }, []);

  const pickFile = () => document.getElementById("fileInput")?.click();

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResultImg(null); setCount(null); setKgHa(null); setError("");
  };

  const run = async () => {
    if (!file) return;
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("conf", String(conf));
      fd.append("iou", String(IOU_FIXED));
      fd.append("imgsz", String(IMGSZ_FIXED));
      fd.append("frame_cm", String(frameCm));
      fd.append("tkw_g", String(tkw));
      const r = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();

      const resImg = data.img_b64 ?? null;
      setCount(data.count ?? null);
      setKgHa(data.kg_ha ?? null);
      setResultImg(resImg);

      // миниатюра + защита от огромных картинок в истории
      const thumb = await makeThumb(resImg, 640, 0.8);
      const resultToStore = resImg && resImg.length > BIG_DATAURL ? null : resImg;

      addHistory({
        ts: Date.now(),
        conf, frameCm, tkw,
        count: data.count ?? null,
        kgHa: data.kg_ha ?? null,
        result: resultToStore,  // может быть null, если слишком большой
        thumb: thumb || resultToStore, // всегда стараемся иметь превью
      });

      setTab("history");
    } catch (e) {
      setError(`Ошибка запроса: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fmt = (ts) => new Date(ts).toLocaleString();

  return (
    <div className="wrap">
      <div className="badge">✨ bid.ai</div>
      <h1 className="title">Подсчёт зёрен и потерь урожая</h1>

      <div className="tabs">
        <button className={`tab ${tab==="analyze"?"active":""}`} onClick={()=>setTab("analyze")}>Анализ</button>
        <button className={`tab ${tab==="history"?"active":""}`} onClick={()=>setTab("history")}>
          История{history.length ? ` (${history.length})` : ""}
        </button>
      </div>

      {tab === "analyze" && (
        <>
          {error && <div className="err card">{error}</div>}

          <div className="card">
            <div ref={dropRef} className="drop" onClick={pickFile}>
              <input id="fileInput" type="file" accept="image/*" onChange={onFileChange}/>
              {file ? "📷 Фото выбрано — можно считать" : "Перетащи или выбери фото рамки 50×50 см"}
            </div>

            {preview && <img src={preview} alt="preview" className="img" />}

            <div className="controls">
              <label>Confidence
                <input type="number" value={conf} min={0.05} max={0.95} step={0.01}
                       onChange={e=>setConf(Number(e.target.value))}/>
              </label>
              <label>Сторона рамки (см)
                <input type="number" value={frameCm} onChange={e=>setFrameCm(Number(e.target.value))}/>
              </label>
              <label>Масса 1000 зёрен (г)
                <input type="number" step={0.5} value={tkw} onChange={e=>setTkw(Number(e.target.value))}/>
              </label>
            </div>

            <button className="btnPrimary" disabled={!file||loading} onClick={run}>
              {loading ? "Рассчитываем…" : "Рассчитать"}
            </button>

            <div className="metrics">
              <div className="metric"><div className="cap">Зёрен (pred)</div><div className="val">{count ?? "—"}</div></div>
              <div className="metric"><div className="cap">Потери (кг/га)</div><div className="val">{kgHa ?? "—"}</div></div>
            </div>
          </div>

          {resultImg && (
            <div className="card">
              <h3 className="sub">Предсказание</h3>
              <img src={resultImg} alt="prediction" className="img" />
              <div className="row" style={{marginTop:12}}>
                <div className="hmeta">
                  <span>conf={conf}</span>
                  <span>imgsz={IMGSZ_FIXED}</span>
                  <span>iou={IOU_FIXED}</span>
                  <span>рамка={frameCm}см</span>
                  <span>tkw={tkw}г</span>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {tab === "history" && (
        <>
          <div className="card" style={{display:"flex", justifyContent:"space-between", alignItems:"center"}}>
            <div className="sub" style={{margin:0}}>История анализов</div>
            {history.length > 0 && (
              <div className="hbtns">
                <button className="btn" onClick={()=>setTab("analyze")}>← К анализу</button>
                <button className="btn" onClick={clearHistory}>Очистить всё</button>
              </div>
            )}
          </div>

          {history.length === 0 && (
            <div className="card">История пуста. Проведи анализ во вкладке «Анализ», и он появится здесь.</div>
          )}

          {history.map((h) => (
            <div key={h.id} className="card hitem">
              <img className="thumb" src={h.thumb || h.result} alt="thumb" />
              <div style={{flex:1}}>
                <div className="hmeta">
                  <strong>{fmt(h.ts)}</strong>
                  <span>conf={h.conf}</span>
                  <span>рамка={h.frameCm}см</span>
                  <span>tkw={h.tkw}г</span>
                  <span>зёрен={h.count ?? "—"}</span>
                  <span>потери={h.kgHa ?? "—"} кг/га</span>
                </div>
                <div className="hbtns">
                  {(h.result || h.thumb) && (
                    <a className="btn btnGold" href={h.result || h.thumb} download={`bidai_pred_${h.id}.png`}>
                      Скачать PNG
                    </a>
                  )}
                  <button
                    className="btn"
                    onClick={() => {
                      if (h.result || h.thumb) setResultImg(h.result || h.thumb);
                      setCount(h.count); setKgHa(h.kgHa);
                      setTab("analyze");
                    }}
                  >
                    Открыть
                  </button>
                  <button className="btn" onClick={()=>removeHistory(h.id)}>Удалить</button>
                </div>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
