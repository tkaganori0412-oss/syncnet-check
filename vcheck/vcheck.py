# 動画の検品（Gemini の代わり・数字で決める・同じ動画なら毎回同じ答え）
# 入力：動画の URL（mp4）・見る種類（raw＝話す動画そのもの／final＝字の入った完成動画）。
# 出力 out/result.json：
#   faces[]   … 4コマ/秒ごとの顔の数（MediaPipe Face Detection・確かさ 0.6 以上）
#   no_face   … 顔が見えない秒・multi_face … 顔が2つ以上の秒
#   jumps[]   … 顔の形の急な変化（Face Mesh の 468 点の動き÷顔の大きさ が 0.12 を超えたコマ＝溶け・ゆがみ・別人になる崩れの合図）
#   mesh_lost … 顔は見えるのに顔の点が取れない秒（形が崩れていると取れない）
#   hands[]   … 手が見えた秒と、手の点の確かさ（MediaPipe Hands）。指の形がおかしいと点の並びが手の形にならない＝bad_hand（指の長さの比が外れる）
#   texts[]   … 1コマ/秒ごとに読めた字（EasyOCR 日本語・英語・確かさ 0.5 以上・2字以上）
#   red_rings[] … 1コマ/秒ごとの赤い輪（中が空いた赤い丸・図形の描き違いの見張り）{t, cx, cy, r}（画面の幅に対する割合）
# 結果の要約は GitHub の注記（::notice）にも出す（Artifacts を取れない所からでも読めるように）
import os, sys, json, subprocess, tempfile, math
import numpy as np, cv2, requests
import mediapipe as mp

URL = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else "raw"
os.makedirs("out", exist_ok=True)
res = {"video_url": URL, "mode": MODE, "error": None}
try:
    tmp = tempfile.mkdtemp(); src = os.path.join(tmp, "v.mp4")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(src, "wb") as f:
            for ch in r.iter_content(1 << 20): f.write(ch)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps
    step = max(1, int(round(fps / 4)))
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hd = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
    faces, jumps, hands, mesh_lost = [], [], [], []
    prev = None; ocr_frames = []
    sec_step = max(1, int(round(fps)))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            t = round(i / fps, 2)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            d = fd.process(rgb)
            dets = d.detections or []
            boxes = [(x.location_data.relative_bounding_box.width * w) for x in dets]
            faces.append({"t": t, "n": len(dets)})
            m = fm.process(rgb)
            if m.multi_face_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in m.multi_face_landmarks[0].landmark])
                size = max(1.0, float(np.ptp(pts[:, 0])))
                if prev is not None:
                    pp, ps = prev
                    c = pts - pts.mean(0); cp = pp - pp.mean(0)
                    mv = float(np.median(np.linalg.norm(c / size - cp / ps, axis=1)))
                    if mv > 0.12: jumps.append({"t": t, "move": round(mv, 3)})
                prev = (pts, size)
            else:
                if dets: mesh_lost.append(t)
                prev = None
            hh = hd.process(rgb)
            if hh.multi_hand_landmarks:
                wl = hh.multi_hand_world_landmarks or [None] * len(hh.multi_hand_landmarks)
                for lm, hd_c, wm in zip(hh.multi_hand_landmarks, hh.multi_handedness or [], wl):
                    sc = float(hd_c.classification[0].score)
                    P = np.array([[p.x * w, p.y * h] for p in lm.landmark])
                    bw = float(np.ptp(P[:, 0])) / w; bh = float(np.ptp(P[:, 1])) / h
                    # 指の長さは手の 3D の点（メートル）で測る＝向きで短く見えるのに引っぱられない
                    Q = np.array([[p.x, p.y, p.z] for p in wm.landmark]) if wm is not None else None
                    fingers = []
                    if Q is not None:
                        for ch in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
                            fingers.append(float(sum(np.linalg.norm(Q[ch[k + 1]] - Q[ch[k]]) for k in range(3))))
                    palm = float(np.linalg.norm(Q[0] - Q[9])) if Q is not None else 0.0
                    big = bw * bh >= 0.004  # 小さい手・腕組みで隠れた手は判定しない
                    bad = bool(big and sc >= 0.8 and fingers and (max(fingers[1:]) > 0.13 or palm > 0.14 or palm < 0.05))
                    hands.append({"t": t, "bad_hand": bad, "score": round(sc, 2), "area": round(bw * bh, 4), "palm": round(palm, 3), "fingers": [round(f, 3) for f in fingers]})
        if i % sec_step == 0: ocr_frames.append((round(i / fps, 2), fr))
        i += 1
    import easyocr
    rd = easyocr.Reader(["ja", "en"], gpu=False, verbose=False)
    def rings(fr):
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV); H, W = fr.shape[:2]
        m = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 90), (180, 255, 255))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = []
        for c in cs:
            hull = cv2.convexHull(c); A = cv2.contourArea(hull); P = cv2.arcLength(hull, True)
            if P < 0.12 * W or A <= 0: continue
            circ = 4 * math.pi * A / (P * P)
            mk = np.zeros(m.shape, np.uint8); cv2.drawContours(mk, [hull], -1, 255, -1)
            fill = float((m[mk > 0] > 0).mean())
            if circ > 0.6 and fill < 0.5:
                (cx, cy), r = cv2.minEnclosingCircle(hull)
                out.append({"cx": round(cx / W, 3), "cy": round(cy / H, 3), "r": round(r / W, 3), "fill": round(fill, 2)})
        return out
    if os.environ.get("FRAMES") == "1":
        # 見本の一覧（1秒1コマ・8枚×1行の横長の絵）を out/sheets に出す（目で確かめる用）
        os.makedirs("out/sheets", exist_ok=True)
        th = [cv2.putText(cv2.resize(fr, (216, 384)), "%ds" % int(t), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) for t, fr in ocr_frames]
        for k in range(0, len(th), 8):
            row = th[k:k + 8] + [np.zeros_like(th[0])] * (8 - len(th[k:k + 8]))
            cv2.imwrite("out/sheets/s%02d.jpg" % (k // 8), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 70])
    red_rings = []
    for t, fr in ocr_frames:
        for rg in rings(fr): red_rings.append({"t": t, **rg})
    texts = []
    for t, fr in ocr_frames:
        small = cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2))
        for box, txt, cf in rd.readtext(small):
            if cf >= 0.5 and len(txt.strip()) >= 2:
                texts.append({"t": t, "text": txt.strip(), "conf": round(float(cf), 2)})
    sec = lambda arr: sorted(set(int(x) for x in arr))
    def runs(ts, gap):
        # 続けて出た区間（コマの間が gap 以下）だけを [始め, 終わり] で返す。1コマだけの見間違いは落とす
        ts = sorted(ts); out = []; st = None; pv = None
        for x in ts:
            if st is None: st = pv = x; continue
            if x - pv <= gap + 1e-6: pv = x; continue
            if pv > st: out.append([st, pv])
            st = pv = x
        if st is not None and pv > st: out.append([st, pv])
        return out
    res.update({"duration": round(dur, 2), "fps": fps, "faces": faces,
                "no_face": sec([f["t"] for f in faces if f["n"] == 0]),
                "multi_face": sec([f["t"] for f in faces if f["n"] >= 2]),
                "jumps": jumps, "mesh_lost": sec(mesh_lost), "hands": hands,
                "bad_hand": sec([x["t"] for x in hands if x["bad_hand"]]),
                "bad_hand_runs": runs([x["t"] for x in hands if x["bad_hand"]], step / fps + 0.01),
                "finger_max": sorted([{"t": x["t"], "f": max(x["fingers"] or [0]), "palm": x["palm"], "a": x["area"], "s": x["score"]} for x in hands], key=lambda z: -z["f"])[:8], "texts": texts,
                "red_rings": red_rings})
except Exception as e:
    res["error"] = str(e)[:500]
json.dump(res, open("out/result.json", "w"), ensure_ascii=False)
summ = {k: v for k, v in res.items() if k not in ("faces", "hands", "video_url", "red_rings", "texts")}
# 要約は短く：大きい赤い輪（画面の幅の 25% 以上＝顔を囲む丸）の秒・字は重ならない分だけ
summ["big_red_ring"] = sorted(set(int(r["t"]) for r in (res.get("red_rings") or []) if r["r"] >= 0.25))
seen = set(); tx = []
for x in (res.get("texts") or []):
    if x["text"] not in seen: seen.add(x["text"]); tx.append([int(x["t"]), x["text"]])
summ["texts_u"] = tx[:60]
summ["hands_n"] = len(res.get("hands") or []); summ["faces_n"] = len(res.get("faces") or [])
js = json.dumps(summ, ensure_ascii=False, separators=(",", ":"))
print(js[:3000])
for k in range(0, min(len(js), 9000), 900):
    print("::notice title=vresult %d::%s" % (k // 900, js[k:k + 900].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")))
