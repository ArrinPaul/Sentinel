# Methodology

How Sentinel decides whether a live person is present: how challenges are generated, how each score is computed, how the scores are combined, what happens when an optional model is missing, and how the signed ledger and tokens work. Everything here is implemented in `backend/app/`.

The numbers below are the constants in the code. Many were tuned by hand and several were deliberately lowered so real users pass more easily. None have been validated against real attacks, so read this as a description of the design, not a claim of accuracy.

[← Back to README](./README.md)

## Contents

1. [Verification flow](#1-verification-flow)
2. [Challenges and anti-replay](#2-challenges-and-anti-replay)
3. [Liveness score](#3-liveness-score)
4. [Deepfake score](#4-deepfake-score)
5. [Emotion score](#5-emotion-score)
6. [Final decision](#6-final-decision)
7. [Sessions and limits](#7-sessions-and-limits)
8. [Token](#8-token)
9. [Audit ledger](#9-audit-ledger)
10. [Limitations](#10-limitations)

---

## 1. Verification flow

1. The browser calls `POST /api/auth/verify` and receives a session ID and a WebSocket address.
2. Over the WebSocket the server sends 3 challenges and a nonce. For each, the browser streams frames and the server runs the challenge check.
3. When the challenges end, the server runs the whole-clip analysis (liveness, emotion, deepfake), combines the scores, saves the result, writes a ledger block and, on success, issues a token.

Frames are subsampled before the heavy analysis: at most 60 frames go into the final pipeline, and liveness uses at most 20 of them.

## 2. Challenges and anti-replay

Each session gets 3 challenges. Each one is a **gesture** or an **expression**, chosen with equal probability by Python's `secrets` module:

| Type | Pool |
| :--- | :--- |
| Gestures (10) | nod up, nod down, turn left, turn right, tilt left, tilt right, open mouth, close eyes, raise eyebrows, blink |
| Expressions (5) | smile, frown, surprised, neutral, angry |

The sequence has a random 32-character hex **nonce** (`secrets.token_hex(16)`) and a timestamp. The nonce is stored so a recorded session cannot be replayed against a new challenge, and expired nonces are purged hourly (24-hour lifetime). The number of possible sequences is small, since there are 15 options for each of 3 slots, about 3,375 combinations, so unpredictability rests mainly on the nonce and on the video having to match the *current* challenge.

**Checking a gesture** uses face-landmark geometry from MediaPipe. For head movements, such as nodding, turning and tilting, the nose, eye and chin landmarks are tracked across frames and the movement in the requested direction must exceed a threshold. For `close_eyes` and `blink`, the blendshape scores from the model are used, with the **eye aspect ratio (EAR)** as a fallback:

$$\mathrm{EAR} = \frac{\lVert p_{\text{top}} - p_{\text{bottom}} \rVert}{\lVert p_{\text{left}} - p_{\text{right}} \rVert}$$

where the points are the eyelid and eye-corner landmarks. A blink shows as a brief drop in EAR. **Checking an expression** uses the emotion analyzer (§5).

## 3. Liveness score

The whole-clip liveness score has two equal parts:

$$L_{\text{clip}} = 0.5\,D + 0.5\,M$$

**Depth cues $D$** come from the 3D landmark coordinates of the first frame with a detected face. A flat photo has little depth variation, a real face has more:

| Cue | Weight | Computed as |
| :--- | :---: | :--- |
| Nose protrusion | 0.50 | $\min(1,\ \lvert z_{\text{nose}} - z_{\text{face plane}} \rvert / 0.03)$ |
| Depth variance | 0.35 | $\min(1,\ \mathrm{Var}(z) / 0.001)$ |
| Width perspective | 0.15 | $\min(1,\ \lvert \text{mouth width} / \text{eye width} - 0.7 \rvert / 0.2)$ |

**Micro-movement $M$** looks for the small involuntary motion of a living face across frames:

| Cue | Weight | Computed as |
| :--- | :---: | :--- |
| Blinking | 0.50 | $\min(1,\ \mathrm{Var}(\mathrm{EAR}) / 0.002)$ |
| Head motion | 0.30 | $\min(1,\ \text{mean frame-to-frame head movement} / 0.003)$ |
| Landmark jitter | 0.20 | $\min(1,\ \text{variance of the nose position} / 0.00003)$ |

If the face landmarker model is not available, only the micro-movement half exists and $L_{\text{clip}} = 0.5\,M$.

**Challenge boost.** After this, the server blends in the challenge success rate $c$ (completed challenges divided by 3):

$$L = \min\bigl(1,\ 0.5\,L_{\text{clip}} + 0.5\,c\bigr)$$

So half of the final liveness score is simply how many challenges the user passed.

## 4. Deepfake score

Two scores are combined, a spatial score $S$ averaged over several frames and a temporal score $T$ from consistency across frames:

$$F = 0.6\,S + 0.4\,T$$

- **With a model file** (`DEEPFAKE_MODEL_PATH`), $S$ comes from a small MesoNet-style convolutional network built in TensorFlow.
- **Without a model**, $S$ is a mix of four classic image checks:

$$S = 0.35\,S_{\text{fft}} + 0.30\,S_{\text{warp}} + 0.20\,S_{\text{color}} + 0.15\,S_{\text{edge}}$$

  These look at frequency patterns typical of generated images, face-warping artifacts, color consistency and edge coherence. On any error the fallback returns a neutral 0.5.

A higher score means more likely to be real. If the clip has no usable frames, the liveness, emotion and deepfake scores are all set to 0. If the temporal analysis fails, $T$ is 0 and the result is only $0.6\,S$, so a failure there lowers the score.

## 5. Emotion score

For the final clip the emotion analyzer scores how natural the expressions look. It uses **DeepFace** to read an emotion per frame, then combines three parts:

$$E = 0.4\,\text{transition} + 0.3\,\text{confidence} + 0.3\,\text{match}$$

- **transition:** starts at 1 and loses points for patterns that look synthetic: 0.15 for each frame pair where the dominant emotion changes while both frames have confidence above 0.7, 0.10 for each pair whose confidence jumps by more than 0.5, 0.25 if every frame has the identical confidence value, and 0.10 if more than 95% of pairs have near-identical confidence. The total penalty is capped at 1. With fewer than two frames it returns 0.7.
- **confidence:** the average confidence of the detected emotions.
- **match:** for an expression challenge, the fraction of frames showing the expected emotion, doubled and capped at 1. It is 1 when no expression is expected.

**If DeepFace is not installed** (it is **not** in `requirements.txt`), the emotion score is a **fixed 0.70**, chosen so this component "doesn't block real users". It then measures nothing, and it contributes $0.35 \times 0.70 = 0.245$ of the 0.65 pass mark (about 38%) no matter who is in front of the camera.

## 6. Final decision

$$\text{final} = 0.40\,L + 0.25\,F + 0.35\,E,\qquad \text{pass} \iff \text{final} \ge 0.65$$

**Worked example (DeepFace missing).** With $E = 0.70$, the other two terms must supply at least $0.65 - 0.245 = 0.405$. A user with liveness $L = 0.80$ and deepfake score $F = 0.5$ gets $0.32 + 0.125 + 0.245 = 0.69$ and passes. A flat, static image would need $0.4L + 0.25F \ge 0.405$, for instance $L = 0.6$ with $F = 0.65$. Because $L$ is half the challenge success rate, a clip that is judged mostly on challenges can score well even when the depth and micro-movement cues are weak. The thresholds have not been tested against spoofing attacks, so how hard this is to fool in practice is unknown.

## 7. Sessions and limits

| Limit | Value |
| :--- | :--- |
| Session duration | 120 seconds |
| Consecutive failed challenges before the session ends | 3 (a success resets the counter) |
| Nonce lifetime | 24 hours |

Sessions, nonces and audit logs are kept in an in-memory store (`database_service.py`) and are lost when the server restarts.

## 8. Token

A successful verification returns a JWT signed with **RS256**:

| Claim | Meaning |
| :--- | :--- |
| `sub` | the user ID |
| `session_id` | the verification session |
| `final_score` | the final score |
| `iat`, `exp` | issued at, and 15 minutes later |
| `iss` | the issuer name |

`POST /api/token/validate` verifies the signature and expiry. If no key pair is supplied through `JWT_PRIVATE_KEY` and `JWT_PUBLIC_KEY`, a 2048-bit RSA pair is generated when the server starts, so tokens do not outlive a restart.

## 9. Audit ledger

Each completed verification (and each token issue) is appended to a **hash-chained log**. A block holds an index, a timestamp, its data, the hash of the previous block, a random nonce and its own hash:

$$h_i = \mathrm{SHA256}\bigl(\text{index},\ \text{time},\ \text{data},\ h_{i-1},\ \text{nonce}\bigr)$$

and an **RSA-PSS signature** (SHA-256) over $h_i$. The first block's previous hash is 64 zeros.

- **Integrity check** recomputes each hash, checks that each block points at the previous one and verifies each signature, so changing an old block breaks the chain after it.
- **Proof endpoint** returns a block with its hash and signature, so anyone with the public key can verify that block alone.
- **Storage:** the chain is saved to `data/verification_ledger.json` and its key to `data/ledger_keys.json`.

There is no mining, no consensus and no second node. The server's operator holds the signing key and could rewrite and re-sign the whole chain, so this proves consistency, not independence. The code comments call it "decentralized", but it is a single-node signed log.

## 10. Limitations

- **Not evaluated against attacks.** Printed photos, phone replays, 3D masks and modern face-swap video have not been tried, and there are no false-accept or false-reject rates.
- **Emotion factor is empty by default** (§5).
- **Challenge success counts for half of liveness** (§3), and there are only 3 challenges.
- **Hand-tuned divisors** (for example 0.002, 0.003, 0.00003) were chosen from rough ranges described in comments and lowered for sensitivity.
- **MediaPipe landmarks are not metric.** Depth values are normalized and noisy, and are not calibrated for camera distance or lens.
- **Single-node, in-memory state.** Sessions disappear on restart and the ledger cannot be independently audited.
- **Privacy.** Webcam frames are processed on the server. The code does not store the frames, but it does store scores and user IDs in the ledger, which its endpoints expose without a login.
