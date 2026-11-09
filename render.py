# 書き出し（Shotstack の代わり）：job の JSON（R2 に置いたもの）に書かれた ffmpeg の命令を順に動かす。
# job = {"steps":[{"cmd":"-i {{in_0}} ... {{out_1}}","inputs":{"in_0":"https://…"},"outputs":{"out_1":"manga.mp4"}}]}
# inputs の値が "out:out_1" なら前の段の出力を使う。出力はすべて out/ に置き、成果物として返す。
import json, os, re, shlex, subprocess, sys, time, requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
job_url = sys.argv[1]
os.makedirs("out", exist_ok=True); os.makedirs("in", exist_ok=True)
res = {"ok": False, "steps": []}
def save():
    json.dump(res, open("out/result.json", "w"), ensure_ascii=False)
def get(u, p):
    for k in range(3):
        try:
            with requests.get(u, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(p, "wb") as f:
                    for c in r.iter_content(1 << 20): f.write(c)
            return
        except Exception as e:
            if k == 2: raise RuntimeError(f"download {u}: {e}")
            time.sleep(2 + k * 3)
try:
    job = requests.get(job_url, timeout=60).json()
    paths, cache = {}, {}
    for si, st in enumerate(job["steps"]):
        t0 = time.time(); todo = []
        for name, u in st.get("inputs", {}).items():
            if u.startswith("out:"): paths[name] = paths[u[4:]]; continue
            if u in cache: paths[name] = cache[u]; continue
            ext = os.path.splitext(urlparse(u).path)[1][:6] or ".bin"
            p = f"in/{name}{ext}"; paths[name] = p; cache[u] = p; todo.append((u, p))
        with ThreadPoolExecutor(8) as ex: list(ex.map(lambda a: get(*a), todo))
        for name, fn in st.get("outputs", {}).items(): paths[name] = "out/" + re.sub(r"[^\w.\-]", "_", fn)
        cmd = re.sub(r"\{\{(\w+)\}\}", lambda m: paths[m.group(1)], st["cmd"])
        t1 = time.time()
        r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + shlex.split(cmd), capture_output=True, text=True)
        res["steps"].append({"step": si, "download_sec": round(t1 - t0, 1), "ffmpeg_sec": round(time.time() - t1, 1), "code": r.returncode, "err": r.stderr[-1500:]})
        save()
        if r.returncode != 0: raise RuntimeError(f"ffmpeg step {si} failed")
    res["ok"] = True
except Exception as e:
    res["error"] = str(e)[:1500]
save()
print(json.dumps(res, ensure_ascii=False)[:3000])
sys.exit(0 if res["ok"] else 1)
