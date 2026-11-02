# SyncNet 検品（口と声のずれを数値にする）
# GitHub Actions 用。入力：動画の URL（mp4）。出力：全体の offset・conf と、1秒ごとの conf・offset、悪い秒の一覧。
import os, sys, json, subprocess, tempfile, types, statistics
import numpy as np, cv2, requests

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "syncnet_python")
MODEL = os.path.join(REPO, "data", "syncnet_v2.model")

def setup():
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/joonson/syncnet_python.git", REPO], check=True)
    if not os.path.exists(MODEL):
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        subprocess.run(["wget", "-q", "-O", MODEL, "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"], check=True)
setup()
sys.path.insert(0, REPO)
from SyncNetInstance import SyncNetInstance  # noqa: E402
NET = SyncNetInstance(device="cpu")
NET.loadParameters(MODEL)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_box(path):
    """顔の位置（カメラ固定なので、何コマかの中央値で1つに決める）"""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = []
    for idx in np.linspace(0, n - 1, 24).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        f = CASCADE.detectMultiScale(g, 1.1, 5, minSize=(int(W * 0.12), int(W * 0.12)))
        if len(f):
            boxes.append(max(f, key=lambda b: b[2] * b[3]))
    cap.release()
    if not boxes:
        raise RuntimeError("顔が見つからない")
    x, y, w, h = [statistics.median([b[i] for b in boxes]) for i in range(4)]
    cx, cy = x + w / 2, y + h / 2 + h * 0.1
    side = max(w, h) * 1.5
    x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
    side = int(min(side, W - x0, H - y0))
    return x0, y0, side, W, H

def check(video_url, bad_conf=3.0):
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src.mp4")
    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(src, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
    x0, y0, side, W, H = face_box(src)
    crop = os.path.join(tmp, "crop.avi")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-vf", f"crop={side}:{side}:{x0}:{y0},scale=224:224,fps=25",
                    "-c:v", "mjpeg", "-q:v", "2", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", crop], check=True)
    opt = types.SimpleNamespace(tmp_dir=tmp, reference="ref", batch_size=20, vshift=15)
    offset, conf, dists = NET.evaluate(opt, crop)          # dists: (コマ, 31)
    d = np.asarray(dists)
    best = int(15 - int(offset))                              # 全体で一番合う列
    med = np.median(d.mean(0))
    per = []
    for s in range(int(len(d) / 25)):
        w = d[s * 25:(s + 1) * 25]
        if len(w) < 10:
            break
        m = w.mean(0)
        per.append({"t": s, "conf": round(float(np.median(m) - m[best]), 3),
                    "local_offset": int(15 - int(np.argmin(m))),
                    "local_conf": round(float(np.median(m) - m.min()), 3)})
    bad = [p for p in per if p["conf"] < float(bad_conf) or abs(p["local_offset"] - int(offset)) >= 3]
    return {"offset": int(offset), "conf": round(float(conf), 3), "frames": int(len(d)),
            "face": {"x": x0, "y": y0, "side": side, "W": W, "H": H},
            "per_sec": per, "bad_sec": [p["t"] for p in bad], "bad_conf": float(bad_conf)}

if __name__ == "__main__":
    url = sys.argv[1]; bad = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 3.0
    try:
        out = check(url, bad)
    except Exception as e:
        out = {"error": str(e)}
    out["video_url"] = url
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/result.json", "w"), ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False))
