#!/usr/bin/env python3
# Joint Token Fusion (A4 accuracy-enabled)
# - Trigger when (len(window)>=K) OR (now-last_infer)>=TIMEOUT_S
# - Decode per enc; softmax(beta); gating w=1
# - CSV logs: p50/p90/p99, cause(K|T), fill_ratio, bytes, pred, gt, correct, acc_running

import os, time, socket, select, argparse, msgpack, csv, pathlib
import numpy as np

def now_s(): return time.time()

def softmax(x, beta=1.0):
    x = np.asarray(x, dtype=np.float32) * float(beta)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def fwht_inplace(a: np.ndarray):
    h = 1; n = a.shape[0]
    while h < n:
        for i in range(0, n, h*2):
            x = a[i:i+h]; y = a[i+h:i+2*h]
            ta = x + y; tb = x - y
            a[i:i+h], a[i+h:i+2*h] = ta, tb
        h *= 2
    return a

# ---------- Decoders ----------
def decode_plain(q_bytes: bytes, meta: dict):
    q = np.frombuffer(q_bytes, dtype=np.int8)
    return q.astype(np.float32) * float(meta["scale"])

def _inv_perm(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv

def decode_psmix(q_bytes: bytes, meta: dict):
    d = int(meta["d"])
    q = np.frombuffer(q_bytes, dtype=np.int8)[:d]
    z = q.astype(np.float32) * float(meta["scale"])
    rng = np.random.RandomState(int(meta["seed"]))
    perm = rng.permutation(d)
    sign = (rng.randint(0,2,size=d)*2 - 1).astype(np.int8)
    invp = _inv_perm(perm)
    zm = z[invp]
    return zm * sign

def decode_hmix(q_bytes: bytes, meta: dict):
    d = int(meta["d"]); n2 = int(meta.get("n2", 1<<(d-1).bit_length()))
    q = np.frombuffer(q_bytes, dtype=np.int8)[:d]
    v = q.astype(np.float32) * float(meta["scale"])
    v = np.pad(v, (0, n2 - d), mode='constant')
    rng = np.random.RandomState(int(meta["seed"]))
    diag = (rng.randint(0,2,size=n2)*2 - 1).astype(np.int8)
    v = fwht_inplace(v.copy()) * (1.0 / n2)
    v = v * diag
    return v[:d]

DECODERS = {"plain": decode_plain, "psmix": decode_psmix, "hmix": decode_hmix}

def percentiles(xs):
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0: return (np.nan, np.nan, np.nan, np.nan)
    return (float(np.mean(xs)),
            float(np.percentile(xs,50)),
            float(np.percentile(xs,90)),
            float(np.percentile(xs,99)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--bind-port", type=int, default=50001)
    ap.add_argument("--window-s", type=float, default=0.20)
    ap.add_argument("--timeout-s", type=float, default=0.05)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--expected-producers", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--csv", default="jtfa_metrics.csv")
    ap.add_argument("--head-npz", default="")  # linear head (C,d) and optional b
    args = ap.parse_args()

    # Linear head
    Wout = None; bout = None
    if args.head_npz and pathlib.Path(args.head_npz).exists():
        with np.load(args.head_npz) as f:
            Wout = f["W"].astype(np.float32)   # (C, d)
            bout = f.get("b", None)
            if bout is not None: bout = bout.astype(np.float32)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.bind_port))
    sock.setblocking(False)

    window = []   # list of dict{z, ts, bytes, producer_id, label}
    last_infer = now_s()
    w = None
    d_dim = None

    acc_num = 0
    acc_den = 0

    with open(args.csv, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["timestamp","cause","m","fill_ratio","bytes",
                     "lat_mean","p50","p90","p99","pred","gt","correct","acc_running"])
        while True:
            r, _, _ = select.select([sock], [], [], args.timeout_s)
            if r:
                while True:
                    try:
                        data, _ = sock.recvfrom(65536)
                    except BlockingIOError:
                        break

                    msg = msgpack.unpackb(data, raw=False)
                    meta = msg["meta"]; enc = meta["enc"]
                    dec = DECODERS[enc]
                    z = dec(msg["q"], meta)
                    label = int(msg.get("label", -1))
                    d_dim = z.shape[-1] if d_dim is None else d_dim
                    if w is None and d_dim is not None:
                        w = np.ones(d_dim, dtype=np.float32)
                    window.append({
                        "z": z.astype(np.float32),
                        "ts": float(msg["ts"]),
                        "bytes": len(data),
                        "producer_id": int(msg.get("producer_id", -1)),
                        "label": label,
                    })

            if d_dim is None:
                d_dim = z.shape[-1]
            else:
                if z.shape[-1] != d_dim:
                    continue 

            pred = -1
            now = now_s()
            cause = None
            if len(window) >= args.min_views:
                cause = "K"
            elif (now - last_infer) >= args.timeout_s:
                cause = "T"

            if cause is not None and len(window) > 0:
                # metrics
                lat = [now - it["ts"] for it in window]
                bsum = sum(it["bytes"] for it in window)
                lat_mean, p50, p90, p99 = percentiles(lat)

                # fusion
                Z = np.stack([it["z"] for it in window], axis=0)  # (m,d)
                scores = Z @ w
                a = softmax(scores, beta=args.beta)
                zstar = a @ Z

                # ground-truth: majority within window (excluding -1)
                labels = [it["label"] for it in window if it["label"] != -1]
                gt = -1
                if len(labels) > 0:
                    vals, cnts = np.unique(labels, return_counts=True)
                    gt = int(vals[int(np.argmax(cnts))])


                # prediction
                if Wout is not None:
                    logits = Wout @ zstar
                    if bout is not None:
                        logits = logits + bout
                    pred = int(np.argmax(logits))

                correct = 0
                if isinstance(pred, int) and gt != -1:
                    correct = int(pred == gt)
                    acc_num += correct
                    acc_den += 1
                acc_running = (float(acc_num) / max(1, acc_den))

                # logging
                m = len(window)
                uniq = len(set([it["producer_id"] for it in window if it["producer_id"] != -1]))
                denom = max(1, args.expected_producers)
                fill_ratio = uniq / denom

                wr.writerow([now, cause, m, f"{fill_ratio:.3f}", bsum,
                             f"{lat_mean:.6f}", f"{p50:.6f}", f"{p90:.6f}", f"{p99:.6f}",
                             pred, gt, correct, f"{acc_running:.6f}"])
                fcsv.flush()
                try:
                    os.fsync(fcsv.fileno())
                except Exception:
                    pass

                window.clear()
                last_infer = now

if __name__ == "__main__":
    main()

