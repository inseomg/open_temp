# open_temp
opensource project in school


# JTFA — Joint Token Fusion Architecture (mini)
엣지 장치들이 **저차원 토큰(d=16)**만 중앙으로 보내고, 중앙 **Fusion**이 **게이팅 기반 가중합**으로 결합해 예측하는 경량 분산 추론 파이프라인입니다.

> 목표: **p99(꼬리)와 tail factor(p99/p50)↓** while **p50/정확도/처리량 유지**, 대역 **≈5.5 kbps** 수준.

---

## 폴더 구조
'''text
.
├─ src/
│  └─ jtfa/
│     ├─ producer.py
│     ├─ fusion.py
│     └─ __init__.py
├─ doc/
│  └─ 오픈소스_프로젝트_제안서_(1반4팀_20222582_전인성).pdf
└─ README.md
'''

--

## 빠른실행 
python src/jtfa/producer.py --demo | python src/jtfa/fusion.py --demo


핵심 아이디어
	•	여러 카메라/드론이 같은 장면을 본다.
	•	네트워크는 가끔 느리거나 끊긴다 → 제일 느린 노드 대기 금지.
	•	시간창 T 동안 도착한 최소 K개 토큰만으로 즉시 추론(K-of-N).
	•	Fusion: 토큰별 **품질 점수 s_i → softmax(β s_i)**로 가중치 계산, 가중합 표현 (\tilde{z}=\sum w_i z_i) → Linear head 예측.

실행 파라미터(예시)

## 실행 파라미터(표)
| 파라미터 | 예시 | 설명 |
|---|---:|---|
| d | 16 | 토큰 차원 |
| r | 5 Hz | 토큰 전송률 |
| T | 0.20 s | 윈도우 길이 |
| K | 3 | 최소 도착 수 |
| W | 0.05 s | 타임아웃 |

대역 근사: kbps ≈ r × d × 4B × 8 / 1000 → d=16, r=5 → ≈ 2.56 kbps (메타 포함 실측 ≈5.5 kbps)

실험 환경 : tmux 와 nmap 을 이용해 연결. 추가로 rpi 4 3대 rpi 5 2대를 이용하여 producer, fusion node 생성하여 진행.
