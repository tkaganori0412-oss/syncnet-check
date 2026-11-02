# 土台の合成：声が無音の時間だけ、口を閉じた土台に差し替える（つなぎ目は重ねる）
import json, os, subprocess, sys, requests
url_norm, url_sil, url_aud = sys.argv[1:4]
min_sil = float(sys.argv[4] or 0.35); fade = float(sys.argv[5] or 0.25)
os.makedirs("out", exist_ok=True)
def get(u, p):
    with requests.get(u, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(p, "wb") as f:
            for c in r.iter_content(1 << 20): f.write(c)
def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode().strip())
def run(a): subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + a, check=True)
get(url_norm, "norm.mp4"); get(url_sil, "sil.mp4"); get(url_aud, "aud.mp3")
A = dur("aud.mp3")
# 無音の区間
log = subprocess.run(["ffmpeg", "-i", "aud.mp3", "-af", f"silencedetect=noise=-35dB:d={min_sil}", "-f", "null", "-"], capture_output=True, text=True).stderr
sil, s0 = [], None
for line in log.splitlines():
    if "silence_start:" in line: s0 = float(line.split("silence_start:")[1].split()[0])
    if "silence_end:" in line and s0 is not None:
        e = float(line.split("silence_end:")[1].split()[0]); sil.append([max(0, s0), min(A, e)]); s0 = None
if s0 is not None: sil.append([s0, A])
# 両方の土台を、声の長さ以上に折り返して伸ばす（同じ大きさ・30fps）
def bounce(src, dst):
    d = dur(src); n = 1
    while d * n < A + 1: n += 1
    run(["-i", src, "-filter_complex", "[0:v]fps=30,scale=720:1264,setsar=1,split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[o]", "-map", "[o]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "fb.mp4"])
    run(["-stream_loop", "-1", "-i", "fb.mp4", "-t", f"{A:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", dst])
bounce("norm.mp4", "n.mp4"); bounce("sil.mp4", "s.mp4")
# 重み（無音の区間の内側で 0→1→0）をコマごとに作り、小さな灰色の動画にして重ね合わせる（速い）
F = 30; nF = int(A * F) + 1
use = [(a, b) for a, b in sil if b - a >= min_sil]
w = []
for k in range(nF):
    t = k / F; v = 0.0
    for a, b in use:
        v = max(v, min(1.0, max(0.0, min((t - a) / fade, (b - t) / fade))))
    w.append(int(round(v * 255)))
open("mask.gray", "wb").write(bytes(sum(([x] * 4 for x in w), [])))
run(["-f", "rawvideo", "-pix_fmt", "gray", "-s", "2x2", "-r", str(F), "-i", "mask.gray", "-i", "n.mp4", "-i", "s.mp4",
     "-filter_complex", "[0:v]scale=720:1264:flags=neighbor,format=gray[m];[2:v]format=yuva420p[sv];[sv][m]alphamerge[sa];[1:v][sa]overlay=0:0:shortest=1,format=yuv420p[o]",
     "-map", "[o]", "-t", f"{A:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-movflags", "+faststart", "out/composite.mp4"])
terms = use
info = {"audio_sec": round(A, 3), "silences": [[round(a, 2), round(b, 2)] for a, b in sil], "used": len(terms), "size": os.path.getsize("out/composite.mp4")}
json.dump(info, open("out/info.json", "w")); print(json.dumps(info))
