# open_temp
opensource project in school


# JTFA — Joint Token Fusion Architecture (mini)
엣지 장치들이 **저차원 토큰(d=16)**만 중앙으로 보내고, 중앙 **Fusion**이 **게이팅 기반 가중합**으로 결합해 예측하는 경량 분산 추론 파이프라인입니다.

> 목표: **p99(꼬리)와 tail factor(p99/p50)↓** while **p50/정확도/처리량 유지**, 대역 **≈5.5 kbps** 수준.

---

## 폴더 구조
```text
.
├─ src/
│  └─ jtfa/
│     ├─ producer.py
│     ├─ fusion.py
│     └─ __init__.py
├─ doc/
│  └─ 오픈소스_프로젝트_제안서_(1반4팀_20222582_전인성).pdf
└─ README.md
