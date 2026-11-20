"""
Naruto Jutsu Hand Tracker
=========================
Gestures:
  - One hand OPEN palm    -> RASENGAN     (spinning blue energy sphere)
  - BOTH hands OPEN       -> RASENSHURIKEN (rasengan + 4 wind blades)
  - FIST                  -> CHIDORI      (crackling lightning)
  - WINK (close one eye)  -> AMATERASU    (black flames engulf the screen)

Controls:
  Q / ESC  - Quit
  R        - Reset effects
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
import numpy as np
import math
import time
import random
import os
import threading
import sounddevice as sd

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
WRIST = 0
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_MCP = 13; RING_PIP = 14; RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

# Eye landmarks for EAR (Eye Aspect Ratio) from MediaPipe FaceMesh 478 points
LEFT_EYE = {'p1': 33, 'p2': 160, 'p3': 158, 'p4': 133, 'p5': 153, 'p6': 144}
RIGHT_EYE = {'p1': 362, 'p2': 385, 'p3': 387, 'p4': 263, 'p5': 373, 'p6': 380}


# ──────────────────────────────────────────────
# Sound Engine (synthesized audio, no files needed)
# ──────────────────────────────────────────────
SAMPLE_RATE = 44100
LOOP_DURATION = 2.0  # seconds per loop buffer

def _gen_rasengan_sound():
    """Spinning energy hum — layered sine waves with amplitude modulation."""
    t = np.linspace(0, LOOP_DURATION, int(SAMPLE_RATE * LOOP_DURATION), endpoint=False)
    # base hum
    sig = 0.25 * np.sin(2 * np.pi * 180 * t)
    sig += 0.15 * np.sin(2 * np.pi * 360 * t)
    sig += 0.10 * np.sin(2 * np.pi * 540 * t)
    # spinning modulation
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 7 * t)
    sig *= mod
    # subtle whoosh noise
    noise = np.random.randn(len(t)) * 0.04
    # simple low-pass via cumulative average
    for i in range(1, len(noise)):
        noise[i] = 0.95 * noise[i-1] + 0.05 * noise[i]
    sig += noise
    # fade the loop edges for seamless looping
    fade = 800
    sig[:fade] *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.6).astype(np.float32)

def _gen_rasenshuriken_sound():
    """Wind shuriken — rasengan base + layered wind noise + higher pitch."""
    t = np.linspace(0, LOOP_DURATION, int(SAMPLE_RATE * LOOP_DURATION), endpoint=False)
    sig = 0.20 * np.sin(2 * np.pi * 220 * t)
    sig += 0.12 * np.sin(2 * np.pi * 440 * t)
    sig += 0.08 * np.sin(2 * np.pi * 880 * t)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * t)
    sig *= mod
    # heavier wind noise (bandpass effect via two cascaded filters)
    wind = np.random.randn(len(t)) * 0.12
    for i in range(1, len(wind)):
        wind[i] = 0.88 * wind[i-1] + 0.12 * wind[i]
    wind -= np.convolve(wind, np.ones(200)/200, mode='same')  # remove DC / very low freq
    sig += wind
    fade = 800
    sig[:fade] *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.6).astype(np.float32)

def _gen_chidori_sound():
    """Electric crackling — amplitude-modulated noise + chirping tones (1000 birds)."""
    t = np.linspace(0, LOOP_DURATION, int(SAMPLE_RATE * LOOP_DURATION), endpoint=False)
    # crackling static
    noise = np.random.randn(len(t)) * 0.3
    # rapid amplitude gating for crackle effect
    gate = np.abs(np.sin(2 * np.pi * 60 * t)) ** 4
    crackle = noise * gate
    # chirping birds — several high sine tones with fast modulation
    birds = np.zeros_like(t)
    for freq in [1200, 1800, 2400, 3000, 3600]:
        amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(8, 20) * t + random.uniform(0, 6))
        birds += 0.03 * np.sin(2 * np.pi * freq * t) * amp_mod
    # electric buzz undertone
    buzz = 0.15 * np.sin(2 * np.pi * 100 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 15 * t))
    sig = crackle + birds + buzz
    fade = 800
    sig[:fade] *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.5).astype(np.float32)

def _gen_amaterasu_sound():
    """Deep ominous rumble — low frequency drone + fire crackle."""
    t = np.linspace(0, LOOP_DURATION, int(SAMPLE_RATE * LOOP_DURATION), endpoint=False)
    # deep drone
    drone = 0.35 * np.sin(2 * np.pi * 45 * t)
    drone += 0.20 * np.sin(2 * np.pi * 90 * t)
    drone += 0.10 * np.sin(2 * np.pi * 135 * t)
    # slow pulsing
    drone *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t)
    # fire crackle — sparse random pops
    crackle = np.zeros_like(t)
    pop_indices = np.random.choice(len(t), size=int(len(t) * 0.003), replace=False)
    crackle[pop_indices] = np.random.uniform(0.3, 0.8, len(pop_indices))
    # smooth the pops slightly
    for i in range(1, len(crackle)):
        crackle[i] = max(crackle[i], crackle[i-1] * 0.97)
    crackle *= np.random.randn(len(t)) * 0.5
    sig = drone + crackle
    fade = 800
    sig[:fade] *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.6).astype(np.float32)


class SoundEngine:
    """Manages looping playback of jutsu sounds with crossfade transitions."""

    def __init__(self):
        print("[INFO] Generating jutsu sound effects...")
        self.buffers = {
            "RASENGAN": _gen_rasengan_sound(),
            "RASENSHURIKEN": _gen_rasenshuriken_sound(),
            "CHIDORI": _gen_chidori_sound(),
            "AMATERASU": _gen_amaterasu_sound(),
        }
        self.active_jutsu = None
        self.target_jutsu = None
        self.position = 0
        self.volume = 0.0          # current volume (for fade in/out)
        self.target_volume = 0.0
        self.fade_speed = 0.002    # per sample
        self._lock = threading.Lock()
        self.stream = None

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=1024,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)

        with self._lock:
            jutsu = self.active_jutsu
            target = self.target_jutsu

        # if we need to switch jutsu and volume is near zero, do the switch
        if jutsu != target and self.volume < 0.01:
            jutsu = target
            with self._lock:
                self.active_jutsu = target
            self.position = 0

        # determine target volume
        if target is None:
            tv = 0.0
        elif jutsu == target:
            tv = 1.0
        else:
            tv = 0.0  # fade out before switching

        buf = self.buffers.get(jutsu) if jutsu else None

        if buf is not None:
            buf_len = len(buf)
            for i in range(frames):
                # smooth volume
                if self.volume < tv:
                    self.volume = min(tv, self.volume + self.fade_speed)
                elif self.volume > tv:
                    self.volume = max(tv, self.volume - self.fade_speed)
                out[i] = buf[self.position % buf_len] * self.volume
                self.position += 1
        else:
            # fade out any residual
            self.volume = max(0.0, self.volume - self.fade_speed)

        outdata[:, 0] = out

    def set_jutsu(self, jutsu_name):
        """Set which jutsu sound should play (None to stop)."""
        with self._lock:
            self.target_jutsu = jutsu_name

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ──────────────────────────────────────────────
# Particle System
# ──────────────────────────────────────────────
class Particle:
    __slots__ = ['x','y','vx','vy','life','max_life','size','color','kind']
    def __init__(self, x, y, vx, vy, life, size, color, kind='energy'):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = self.max_life = life
        self.size, self.color, self.kind = size, color, kind
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.life -= 1
        self.vx *= 0.97; self.vy *= 0.97
    @property
    def alive(self): return self.life > 0
    @property
    def alpha(self): return max(0.0, self.life / self.max_life)


class ParticleSystem:
    def __init__(self, cap=800):
        self.particles: list[Particle] = []
        self.cap = cap

    def _add(self, p):
        if len(self.particles) < self.cap:
            self.particles.append(p)

    # ── Rasengan particles (blue swirl) ──
    def emit_rasengan(self, cx, cy, radius, intensity=1.0):
        for _ in range(int(12 * intensity)):
            a = random.uniform(0, 2*math.pi)
            d = random.uniform(0, radius*1.3)
            s = random.uniform(1.5, 4.0)
            ta = a + math.pi/2 + random.uniform(-0.3, 0.3)
            self._add(Particle(
                cx+math.cos(a)*d, cy+math.sin(a)*d,
                math.cos(ta)*s+random.uniform(-0.5,0.5),
                math.sin(ta)*s+random.uniform(-0.5,0.5),
                random.randint(10,25), random.uniform(1.5,4.0),
                (random.randint(200,255), random.randint(180,255), random.randint(100,200)),
                'energy'))

    # ── Rasenshuriken particles (wind blades) ──
    def emit_rasenshuriken(self, cx, cy, radius, angle_off, intensity=1.0):
        for _ in range(int(20 * intensity)):
            bi = random.randint(0, 3)
            ba = angle_off + (math.pi/2)*bi
            d = random.uniform(radius*0.5, radius*2.5)
            sp = random.uniform(2.0, 6.0)
            self._add(Particle(
                cx+math.cos(ba+random.uniform(-0.15,0.15))*d,
                cy+math.sin(ba+random.uniform(-0.15,0.15))*d,
                math.cos(ba)*sp+random.uniform(-1,1),
                math.sin(ba)*sp+random.uniform(-1,1),
                random.randint(8,20), random.uniform(1.5,3.5),
                (random.randint(180,255), random.randint(220,255), random.randint(150,220)),
                'wind'))
        self.emit_rasengan(cx, cy, radius*0.6, intensity*0.5)

    # ── Chidori particles (electric sparks) ──
    def emit_chidori(self, cx, cy, radius, intensity=1.0):
        for _ in range(int(15 * intensity)):
            a = random.uniform(0, 2*math.pi)
            d = random.uniform(0, radius*1.5)
            s = random.uniform(3.0, 7.0)
            self._add(Particle(
                cx+math.cos(a)*d, cy+math.sin(a)*d,
                math.cos(a)*s+random.uniform(-2,2),
                math.sin(a)*s+random.uniform(-2,2),
                random.randint(5,15), random.uniform(1.0,3.0),
                # electric blue-white palette
                (random.randint(220,255), random.randint(200,255), random.randint(230,255)),
                'lightning'))

    # ── Amaterasu particles (black flames) ──
    def emit_amaterasu(self, cx, cy, w, h, intensity=1.0):
        for _ in range(int(25 * intensity)):
            # flames spawn from bottom and sides, rising up
            px = random.gauss(cx, w*0.3)
            py = random.uniform(h*0.4, h)
            vx = random.uniform(-1.5, 1.5)
            vy = random.uniform(-4.0, -1.0)  # flames rise
            # black/dark purple/dark red palette
            palette = random.choice([
                (10, 0, 10),       # near black
                (30, 0, 40),       # dark purple
                (20, 0, 60),       # dark violet
                (10, 10, 80),      # dark crimson
                (5, 5, 5),         # pure black
                (40, 0, 20),       # dark magenta
            ])
            self._add(Particle(px, py, vx, vy,
                random.randint(20,50), random.uniform(4.0,12.0),
                palette, 'flame'))

    def update(self):
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, frame):
        overlay = frame.copy()
        for p in self.particles:
            a = p.alpha
            sz = max(1, int(p.size * (a if p.kind != 'flame' else (0.5+0.5*a))))
            color = tuple(int(c * a) for c in p.color)
            if p.kind == 'lightning':
                # draw small jagged line instead of circle for sparks
                ex = int(p.x + random.uniform(-6,6))
                ey = int(p.y + random.uniform(-6,6))
                cv2.line(overlay, (int(p.x), int(p.y)), (ex, ey), color, max(1, sz//2))
            elif p.kind == 'flame':
                cv2.circle(overlay, (int(p.x), int(p.y)), sz, color, -1)
            else:
                cv2.circle(overlay, (int(p.x), int(p.y)), sz, color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def clear(self):
        self.particles.clear()


# ──────────────────────────────────────────────
# Jutsu Visual Effects
# ──────────────────────────────────────────────
def draw_rasengan(frame, cx, cy, radius, t):
    """Spinning blue energy sphere."""
    for i in range(3, 0, -1):
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), int(radius+i*12), (255,200,100), -1)
        cv2.addWeighted(ov, 0.08*i, frame, 1-0.08*i, 0, frame)
    for r_s in range(int(radius), 0, -2):
        ratio = r_s / radius
        cv2.circle(frame, (cx,cy), r_s,
                   (int(255*ratio), int(200*ratio+55*(1-ratio)), int(80*ratio+175*(1-ratio))), 1)
    for i in range(5):
        a0 = t*4 + (2*math.pi/5)*i
        pts = []
        for s in range(20):
            a = a0 + s*0.35
            r = radius*(s/20.0)*0.85
            pts.append((int(cx+math.cos(a)*r), int(cy+math.sin(a)*r)))
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32).reshape(-1,1,2)],
                          False, (255,255,255), 1, cv2.LINE_AA)
    cr = max(3, int(radius*0.25 + 2*math.sin(t*6)))
    cv2.circle(frame, (cx,cy), cr, (255,255,255), -1)
    cv2.putText(frame, "RASENGAN!", (cx-60, cy-int(radius)-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,220,100), 2, cv2.LINE_AA)


def draw_rasenshuriken(frame, cx, cy, radius, t):
    """Rasengan core + 4 spinning wind blades."""
    draw_rasengan(frame, cx, cy, int(radius*0.45), t)
    bl = radius*2.2; bw = radius*0.6; rot = t*3
    for i in range(4, 0, -1):
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), int(bl+i*8), (220,255,180), 1)
        cv2.addWeighted(ov, 0.05*i, frame, 1-0.05*i, 0, frame)
    for bi in range(4):
        ba = rot + (math.pi/2)*bi
        tip = (int(cx+math.cos(ba)*bl), int(cy+math.sin(ba)*bl))
        pa = ba+math.pi/2; hw = bw*0.4
        md = bl*0.5; mx = cx+math.cos(ba)*md; my = cy+math.sin(ba)*md
        p1 = (int(mx+math.cos(pa)*hw), int(my+math.sin(pa)*hw))
        p2 = (int(mx-math.cos(pa)*hw), int(my-math.sin(pa)*hw))
        bx = int(cx+math.cos(ba)*radius*0.4); by = int(cy+math.sin(ba)*radius*0.4)
        bpts = np.array([[bx,by],[p1[0],p1[1]],[tip[0],tip[1]],[p2[0],p2[1]]], np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [bpts], (200,255,200))
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [bpts], True, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "RASENSHURIKEN!!", (cx-100, cy-int(bl)-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2, cv2.LINE_AA)


def draw_chidori(frame, cx, cy, radius, t):
    """Crackling lightning / 1000 birds."""
    # electric glow
    for i in range(4, 0, -1):
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), int(radius+i*15), (255,230,200), -1)
        cv2.addWeighted(ov, 0.06*i, frame, 1-0.06*i, 0, frame)

    # core electric ball (white-blue)
    for r_s in range(int(radius), 0, -3):
        ratio = r_s / radius
        b = int(255*ratio)
        g = int(220*ratio + 35*(1-ratio))
        r_val = int(255*ratio)
        cv2.circle(frame, (cx,cy), r_s, (b, g, r_val), 1)

    # lightning bolts shooting outward
    num_bolts = 6 + int(3*math.sin(t*8))
    for i in range(num_bolts):
        angle = t*2.5 + (2*math.pi/num_bolts)*i + random.uniform(-0.3, 0.3)
        # jagged bolt
        px, py = cx, cy
        segments = random.randint(4, 8)
        bolt_pts = [(px, py)]
        for s in range(segments):
            step = radius * (0.3 + 0.25*s) + random.uniform(-5, 5)
            jitter_angle = angle + random.uniform(-0.5, 0.5)
            px = int(px + math.cos(jitter_angle) * step * 0.5)
            py = int(py + math.sin(jitter_angle) * step * 0.5)
            bolt_pts.append((px, py))
        # draw bolt with glow
        pts_arr = np.array(bolt_pts, np.int32).reshape(-1,1,2)
        # outer glow
        cv2.polylines(frame, [pts_arr], False, (255,200,150), 3, cv2.LINE_AA)
        # inner bright
        cv2.polylines(frame, [pts_arr], False, (255,255,255), 1, cv2.LINE_AA)

    # bright pulsing core
    pulse = max(4, int(radius*0.3 + 4*math.sin(t*10)))
    cv2.circle(frame, (cx,cy), pulse, (255,255,255), -1)

    # flicker effect - random bright flash
    if random.random() < 0.3:
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), int(radius*1.2), (255,255,255), -1)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    cv2.putText(frame, "CHIDORI!!", (cx-60, cy-int(radius)-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,230,180), 2, cv2.LINE_AA)


def draw_amaterasu(frame, t, intensity):
    """Black flames consuming the screen."""
    h, w = frame.shape[:2]

    # darken the entire frame progressively
    darkness = min(0.7, intensity * 0.7)
    dark_overlay = np.zeros_like(frame)
    cv2.addWeighted(dark_overlay, darkness, frame, 1-darkness, 0, frame)

    # black flame columns rising from bottom
    num_flames = 12 + int(8 * intensity)
    for i in range(num_flames):
        fx = int((i / num_flames) * w + random.uniform(-30, 30))
        flame_h = int(h * (0.3 + 0.5 * intensity) * random.uniform(0.5, 1.0))
        fy = h  # start from bottom

        # draw flame column as overlapping circles rising up
        flame_pts = []
        for s in range(0, flame_h, 8):
            jx = fx + int(math.sin(t*3 + i + s*0.05) * 15 * random.uniform(0.5, 1.5))
            jy = fy - s
            size = int((6 + 8*intensity) * (1.0 - s/flame_h) * random.uniform(0.6, 1.4))
            if size < 1:
                continue
            # black-purple-dark red colors
            palette = random.choice([
                (5, 0, 5),
                (15, 0, 25),
                (10, 0, 40),
                (20, 5, 50),
                (8, 0, 8),
                (0, 0, 0),
            ])
            cv2.circle(frame, (jx, jy), size, palette, -1)

        # occasional bright purple edge on flames (the "burning" edge)
        if random.random() < 0.4:
            edge_y = fy - flame_h + random.randint(0, max(1, flame_h//3))
            edge_x = fx + int(math.sin(t*5 + i) * 10)
            cv2.circle(frame, (edge_x, edge_y), random.randint(2, 5),
                       (80, 0, 120), -1)

    # "heat shimmer" distortion at edges via random dark patches
    for _ in range(int(20 * intensity)):
        rx = random.randint(0, w-1)
        ry = random.randint(0, h-1)
        rs = random.randint(2, 8)
        if random.random() < 0.7:
            cv2.circle(frame, (rx, ry), rs, (5, 0, 10), -1)

    # Mangekyo Sharingan eye indicator in top-right
    eye_cx, eye_cy, eye_r = w-60, 90, 25
    # red iris
    cv2.circle(frame, (eye_cx, eye_cy), eye_r, (0, 0, 180), -1)
    # black pupil
    cv2.circle(frame, (eye_cx, eye_cy), 8, (0, 0, 0), -1)
    # 3 tomoe (simplified as dots for mangekyo)
    for i in range(3):
        ta = t*2 + (2*math.pi/3)*i
        tx = int(eye_cx + math.cos(ta)*15)
        ty = int(eye_cy + math.sin(ta)*15)
        cv2.circle(frame, (tx, ty), 4, (0, 0, 0), -1)
    # outer ring
    cv2.circle(frame, (eye_cx, eye_cy), eye_r, (0, 0, 100), 2)

    cv2.putText(frame, "AMATERASU!!!", (w//2-120, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 0, 80), 3, cv2.LINE_AA)
    cv2.putText(frame, "AMATERASU!!!", (w//2-120, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 0, 150), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Gesture / Eye Detection
# ──────────────────────────────────────────────
def detect_hand_gesture(landmarks):
    curled = 0
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
        if landmarks[tip].y > landmarks[pip].y:
            curled += 1
    if curled >= 3:
        return 'fist'
    elif curled <= 1:
        return 'open'
    return 'none'


def get_palm_center(lm, w, h):
    ids = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return (int(sum(lm[i].x for i in ids)/5*w),
            int(sum(lm[i].y for i in ids)/5*h))


def get_hand_size(lm, w, h):
    dx = (lm[MIDDLE_TIP].x - lm[WRIST].x)*w
    dy = (lm[MIDDLE_TIP].y - lm[WRIST].y)*h
    return math.sqrt(dx*dx + dy*dy)


def draw_hand_skeleton(frame, lm, w, h):
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80,80,80), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 2, (50,50,50), -1)


def compute_ear(face_lm, eye_indices, w, h):
    """Compute Eye Aspect Ratio."""
    def pt(idx):
        l = face_lm[idx]
        return np.array([l.x * w, l.y * h])
    p1 = pt(eye_indices['p1'])
    p2 = pt(eye_indices['p2'])
    p3 = pt(eye_indices['p3'])
    p4 = pt(eye_indices['p4'])
    p5 = pt(eye_indices['p5'])
    p6 = pt(eye_indices['p6'])
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return 0.3
    return (vert1 + vert2) / (2.0 * horiz)


def detect_wink(face_landmarks, w, h):
    """
    Returns True if exactly one eye is closed (wink).
    """
    lm = face_landmarks
    left_ear = compute_ear(lm, LEFT_EYE, w, h)
    right_ear = compute_ear(lm, RIGHT_EYE, w, h)

    closed_threshold = 0.18
    open_threshold = 0.22

    left_closed = left_ear < closed_threshold
    right_closed = right_ear < closed_threshold
    left_open = left_ear > open_threshold
    right_open = right_ear > open_threshold

    # one closed, other open = wink
    return (left_closed and right_open) or (right_closed and left_open)


# ──────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────
def draw_hud(frame, jutsu_name, charge_level):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,50), (20,20,20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "NARUTO JUTSU TRACKER", (15,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2, cv2.LINE_AA)

    colors = {
        "RASENGAN": (0,255,255),
        "RASENSHURIKEN": (0,255,150),
        "CHIDORI": (255,230,180),
        "AMATERASU": (50,0,180),
    }
    if jutsu_name:
        c = colors.get(jutsu_name, (255,255,255))
        cv2.putText(frame, f"JUTSU: {jutsu_name}", (w-350, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)

    # chakra bar
    by = h-30; bw = 250; bh = 18; bx = 15
    ov2 = frame.copy()
    cv2.rectangle(ov2, (bx,by), (bx+bw,by+bh), (40,40,40), -1)
    cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
    filled = int(bw * min(charge_level, 1.0))
    if filled > 0:
        bar_color = colors.get(jutsu_name, (255,200,100))
        cv2.rectangle(frame, (bx,by), (bx+filled,by+bh), bar_color, -1)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (200,200,200), 1)
    cv2.putText(frame, "CHAKRA", (bx+bw+10, by+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    iy = h-55
    cv2.putText(frame, "PALM=Rasengan | 2 PALMS=Rasenshuriken | FIST=Chidori | WINK=Amaterasu | Q=Quit",
                (15, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(__doc__)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    hand_model = os.path.join(script_dir, "hand_landmarker.task")
    face_model = os.path.join(script_dir, "face_landmarker.task")
    for path, name in [(hand_model, "hand_landmarker.task"), (face_model, "face_landmarker.task")]:
        if not os.path.exists(path):
            print(f"[ERROR] Model not found: {name}")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # MediaPipe Tasks
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        num_hands=2, min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6, running_mode=RunningMode.VIDEO))

    face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model),
        num_faces=1, min_face_detection_confidence=0.6,
        min_tracking_confidence=0.5, running_mode=RunningMode.VIDEO))

    particles = ParticleSystem(cap=1000)
    sound = SoundEngine()
    sound.start()
    start_time = time.time()
    charge = 0.0
    amaterasu_intensity = 0.0
    frame_ts = 0

    window_name = "Naruto Jutsu Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 100, 50)

    print("[INFO] Starting webcam... Press Q or ESC to quit.")

    # warm up
    for _ in range(5):
        ret, f = cap.read()
        if ret:
            cv2.imshow(window_name, cv2.flip(f, 1))
            cv2.waitKey(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t = time.time() - start_time
        frame_ts += 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Hand detection ──
        hand_results = hand_lm.detect_for_video(mp_img, frame_ts)
        gestures = []
        hand_data = []
        if hand_results.hand_landmarks:
            for hlm in hand_results.hand_landmarks:
                draw_hand_skeleton(frame, hlm, w, h)
                g = detect_hand_gesture(hlm)
                cx, cy = get_palm_center(hlm, w, h)
                sz = get_hand_size(hlm, w, h)
                gestures.append(g)
                hand_data.append((g, cx, cy, sz))

        # ── Face / eye detection ──
        face_results = face_lm.detect_for_video(mp_img, frame_ts)
        winking = False
        if face_results.face_landmarks:
            for flm in face_results.face_landmarks:
                winking = detect_wink(flm, w, h)
                if winking:
                    break

        # ── Determine jutsu ──
        num_open = sum(1 for g in gestures if g == 'open')
        num_fist = sum(1 for g in gestures if g == 'fist')

        current_jutsu = None
        if winking:
            current_jutsu = "AMATERASU"
        elif num_open >= 2:
            current_jutsu = "RASENSHURIKEN"
        elif num_open == 1:
            current_jutsu = "RASENGAN"
        elif num_fist >= 1:
            current_jutsu = "CHIDORI"

        # charge
        if current_jutsu:
            charge = min(1.0, charge + 0.05)
        else:
            charge = max(0.0, charge - 0.03)

        # amaterasu builds / fades independently for persistent flames
        if current_jutsu == "AMATERASU":
            amaterasu_intensity = min(1.0, amaterasu_intensity + 0.03)
        else:
            amaterasu_intensity = max(0.0, amaterasu_intensity - 0.015)

        active = current_jutsu if charge > 0.15 else None

        # ── Update sound ──
        sound_jutsu = active or ("AMATERASU" if amaterasu_intensity > 0.05 else None)
        sound.set_jutsu(sound_jutsu)

        # ── Render ──
        # Amaterasu draws first (background effect)
        if amaterasu_intensity > 0.05:
            draw_amaterasu(frame, t, amaterasu_intensity)
            particles.emit_amaterasu(w//2, h//2, w, h, amaterasu_intensity * 0.3)

        if active == "RASENSHURIKEN" and len(hand_data) >= 2:
            acx = int(sum(d[1] for d in hand_data)/len(hand_data))
            acy = int(sum(d[2] for d in hand_data)/len(hand_data))
            asz = sum(d[3] for d in hand_data)/len(hand_data)
            r = int(asz*0.7*charge)
            draw_rasenshuriken(frame, acx, acy, r, t)
            particles.emit_rasenshuriken(acx, acy, r, t*3, charge)

        elif active == "RASENGAN":
            for g, cx, cy, sz in hand_data:
                if g == 'open':
                    r = int(sz*0.35*charge)
                    draw_rasengan(frame, cx, cy, r, t)
                    particles.emit_rasengan(cx, cy, r, charge)
                    break

        elif active == "CHIDORI":
            for g, cx, cy, sz in hand_data:
                if g == 'fist':
                    r = int(sz*0.35*charge)
                    draw_chidori(frame, cx, cy, r, t)
                    particles.emit_chidori(cx, cy, r, charge)
                    break

        particles.update()
        particles.draw(frame)
        draw_hud(frame, active or ("AMATERASU" if amaterasu_intensity > 0.05 else None), charge)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            particles.clear()
            charge = 0.0
            amaterasu_intensity = 0.0
            sound.set_jutsu(None)

    cap.release()
    cv2.destroyAllWindows()
    sound.stop()
    hand_lm.close()
    face_lm.close()
    print("[INFO] Done. Believe it!")


if __name__ == "__main__":
    main()
