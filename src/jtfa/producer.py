# NOTE: assignment requirement — modified on <오늘날짜>


#!/usr/bin/env python3
# Joint Token Producer (A4 label-enabled)
# - Token source: rand | replay(.npy + .labels.txt) | mobilenet
# - Encode: plain | psmix | hmix (int8 quant + scale)
# - Sends UDP msgpack: {ts, producer_id, mode, d, meta, q, label}

import os, time, socket, argparse, msgpack, pathlib
import numpy as np

# Optional torch path (mobilenet feature -> PCA)
try:
    import torch
    import torchvision
    import torchvision.transforms as T
    TORCH_OK = True
except Exception:
    TORCH_OK = False

def now_s() -> float:
    return time.time()

# ----------------------------- Encoding helpers -----------------------------
def quantize_int8(z: np.ndarray):
    s = float(np.max(np.abs(z)) + 1e-8) / 127.0
    q = np.clip(np.round(z / s), -127, 127).astype(np.int8)
    return q, s

def encode_plain(z: np.ndarray, seed: int):
    q, s = quantize_int8(z)
    meta = {"enc": "plain", "seed": seed, "scale": s}
    return q.tobytes(), meta

def _perm_and_sign(d: int, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(d)
    sign = (rng.randint(0, 2, size=d) * 2 - 1).astype(np.int8)  # ±1
    return perm, sign

def encode_psmix(z: np.ndarray, seed: int):
    d = z.shape[-1]
    perm, sign = _perm_and_sign(d, seed)
    zm = z * sign
    zp = zm[perm]
    q, s = quantize_int8(zp)
    meta = {"enc": "psmix", "seed": seed, "scale": s, "d": d}
    return q.tobytes(), meta

def fwht_inplace(a: np.ndarray):
    h = 1; n = a.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i+h]; y = a[i+h:i+2*h]
            ta = x + y; tb = x - y
            a[i:i+h], a[i+h:i+2*h] = ta, tb
        h *= 2
    return a

def encode_hmix(z: np.ndarray, seed: int):
    d = z.shape[-1]
    n2 = 1 << (d - 1).bit_length()  # next pow2
    pad = 0 if n2 == d else (n2 - d)
    v = np.pad(z, (0, pad), mode='constant')
    rng = np.random.RandomState(seed)
    diag = (rng.randint(0, 2, size=n2) * 2 - 1).astype(np.int8)  # ±1
    v = v * diag
    v = fwht_inplace(v.copy()).astype(np.float32)
    v = v[:d]
    q, s = quantize_int8(v)
    meta = {"enc": "hmix", "seed": seed, "scale": s, "d": d, "n2": n2}
    return q.tobytes(), meta

ENCODERS = {"plain": encode_plain, "psmix": encode_psmix, "hmix": encode_hmix}

# ----------------------------- Token sources -----------------------------
def src_rand(d: int):
    """Infinite random tokens; label = -1 (unknown)"""
    while True:
        yield np.random.randn(d).astype(np.float32), -1

def src_replay(np_path: str, labels_path: str):
    """Infinite replay with labels."""
    Z = np.load(np_path).astype(np.float32)   # (N, d)
    N = Z.shape[0]
    labels = None
    if labels_path and pathlib.Path(labels_path).exists():
        with open(labels_path, "r") as f:
            labels = [int(x.strip()) for x in f]
        if len(labels) != N:
            print(f"[WARN] replay labels size {len(labels)} != {N}; using -1")
            labels = None
    idx = 0
    while True:
        z = Z[idx]
        y = labels[idx] if labels is not None else -1
        yield z, y
        idx = (idx + 1) % N

def src_mobilenet(backbone: str, pca_npz: str, img_dir: str, d: int):
    """Infinite mobilenet features → PCA(d); label inferred from filename 'img_<cls>_XXXX.ext'."""
    if not TORCH_OK:
        raise RuntimeError("torch/torchvision not available. Use --token-source replay or rand.")
    if backbone == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").features.eval()
    else:
        model = torch.jit.load(backbone).eval()
    tx = T.Compose([T.Resize(256), T.CenterCrop(224),
                    T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])])
    with np.load(pca_npz) as f:
        mu   = f["mu"].astype(np.float32).squeeze()      # (Df,)
        Wpca = f["Wpca"].astype(np.float32)              # (Df, d)
    paths = sorted([str(p) for p in pathlib.Path(img_dir).glob("*.jpg")] +
                   [str(p) for p in pathlib.Path(img_dir).glob("*.png")])
    if not paths:
        raise RuntimeError(f"No images in {img_dir}")

    def infer_label(p):
        name = pathlib.Path(p).name
        try:
            if name.startswith("img_"):
                return int(name.split("_")[1])
        except Exception:
            pass
        return -1

    labels = [infer_label(p) for p in paths]
    device = torch.device("cpu")
    model.to(device)
    from PIL import Image
    with torch.no_grad():
        while True:
            for p, y in zip(paths, labels):
                x = tx(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                f = model(x).mean([2,3]).cpu().numpy().reshape(-1)   # (Df,)
                z = ((f - mu) @ Wpca).astype(np.float32)             # (d,)
                yield z, y

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50001)
    ap.add_argument("--producer-id", type=int, default=1)
    ap.add_argument("--mode", choices=("plain","psmix","hmix"), default="hmix")
    ap.add_argument("--token-source", choices=("rand","replay","mobilenet"), default="replay")
    ap.add_argument("--replay-npy", default="")
    ap.add_argument("--replay-labels", default="", help="labels.txt (same length as replay npy)")
    ap.add_argument("--mobilenet-backbone", default="mobilenet_v2")
    ap.add_argument("--pca-npz", default="pca.npz")
    ap.add_argument("--img-dir", default="frames")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--rate", type=float, default=10.0)  # tokens/sec
    ap.add_argument("--psmix-seed-base", type=int, default=5000)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.dest, args.port)

    # generator: yields (z, label)
    if args.token_source == "rand":
        gen = src_rand(args.dim)
    elif args.token_source == "replay":
        if not args.replay_npy:
            raise RuntimeError("--replay-npy required for token-source=replay")
        gen = src_replay(args.replay_npy, args.replay_labels)
    else:
        gen = src_mobilenet(args.mobilenet_backbone, args.pca_npz, args.img_dir, args.dim)

    enc_fn = ENCODERS[args.mode]
    seed_counter = 0
    period = 1.0 / max(1e-6, args.rate)

    for z, label in gen:
        seed = args.psmix_seed_base + seed_counter
        q_bytes, meta = enc_fn(z, seed=seed)
        pkt = {
            "ts": now_s(),
            "producer_id": args.producer_id,
            "mode": args.mode,
            "d": int(z.shape[-1]),
            "meta": meta,
            "q": q_bytes,
            "label": int(label),               # <--- label attached
        }
        data = msgpack.packb(pkt, use_bin_type=True)
        sock.sendto(data, addr)
        seed_counter += 1
        time.sleep(period)

if __name__ == "__main__":
    main()
