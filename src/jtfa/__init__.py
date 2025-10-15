# src/jtfa/__init__.py
"""
JTFA (mini) public API

- __version__: 패키지 버전 문자열
- DEFAULTS: producer/fusion 공용 기본값
- get_logger(): 일관된 로거
- mean_fuse(): 간단 토큰 융합(평균)
- demo(): 토큰 N개 생성→평균 융합 데모

Tip) 과제 채점자용 빠른 확인:
    >>> from jtfa import demo; demo()
"""

from __future__ import annotations
from typing import Iterable, Sequence, List, Optional, TypedDict
import os, logging, random

# -------- version --------
try:
    from importlib.metadata import version, PackageNotFoundError  # py>=3.8
    try:
        __version__ = version("jtfa")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

# -------- shared defaults (producer/fusion에서 재사용 가능) --------
DEFAULTS = {
    "BIND_IP":   "0.0.0.0",
    "BIND_PORT": 50001,
    "WINDOW_S":  0.10,
    "TIMEOUT_S": 0.02,
    "MIN_VIEWS": 1,
}

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

# -------- logging --------
def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name or "jtfa")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level if level is not None else logging.INFO)
    return logger

# -------- minimal types & ops --------
class Token(TypedDict):
    id: int
    emb: List[float]

def mean_fuse(vectors: Iterable[Sequence[float]]) -> List[float]:
    """길이 동일한 벡터들의 산술평균. 빈 입력이면 []."""
    vecs = [list(v) for v in vectors]
    if not vecs:
        return []
    d = len(vecs[0])
    # 길이 불일치 방어: 잘못된 벡터는 스킵
    ok = [v for v in vecs if len(v) == d]
    if not ok:
        return []
    return [sum(v[j] for v in ok) / len(ok) for j in range(d)]

def demo(n: int = 8, d: int = 4, seed: Optional[int] = 0) -> None:
    """가벼운 로컬 데모: 무작위 토큰 n개 생성→mean_fuse 결과 출력."""
    rng = random.Random(seed)
    toks: List[Token] = []
    for i in range(n):
        toks.append({"id": i, "emb": [rng.random() for _ in range(d)]})
    fused = mean_fuse([t["emb"] for t in toks])
    print(f"[jtfa.demo] n={n}, d={d}")
    print(" first token:", toks[0])
    print(" fused:", fused)

__all__ = ["__version__", "DEFAULTS", "env_int", "env_float",
           "get_logger", "Token", "mean_fuse", "demo"]
