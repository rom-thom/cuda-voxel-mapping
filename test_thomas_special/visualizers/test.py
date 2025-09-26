#!/usr/bin/env python3
"""
ORCA-style collision checking visualizer (Pygame)

What this does
--------------
- Simulates a 2D environment with circular obstacles.
- Implements a clearance-based collision checker using a 1‑Lipschitz certificate
  (the algorithm you posted).
- Animates the algorithm’s steps: current point, clearance circle, proposed big step,
  certificates, and bisection search on collision.

Controls
--------
Left click & drag        : Move start (P1)
Right click & drag       : Move goal (P2)
Middle click             : Add a circular obstacle at mouse (drag to set radius)
Delete (while hovering)  : Remove obstacle under mouse
Mouse wheel              : Change robot radius
[ / ]                    : Decrease / Increase k (look-ahead multiplier)
A                        : Toggle auto-run
Space                    : Run once from scratch
R                        : Randomize obstacles
C                        : Clear all obstacles
S                        : Snap a screenshot (PNG)
Esc / Q                  : Quit

how long the lines/circles linger:
    . increase linger (things stay longer)
    , decrease linger
    / reset linger to default (about ~0.4s at 90 FPS)

Requirements
------------
pip install pygame numpy

Run
---
python orca_visualizer.py
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pygame

# ----------------------- Config -----------------------

W, H = 900, 650
BG = (19, 21, 26)
GRID = (34, 38, 46)
WHITE = (240, 240, 240)
RED = (230, 70, 70)
GREEN = (60, 190, 120)
BLUE = (80, 150, 255)
YELLOW = (235, 200, 90)
CYAN = (90, 220, 220)
MAGENTA = (220, 120, 220)
GREY = (120, 130, 140)
UI = (200, 200, 210)

FPS = 90

# ----------------------- Geometry helpers -----------------------

def clamp(x, a, b):
    return a if x < a else b if x > b else x

def length(v):
    return math.hypot(v[0], v[1])

# ----------------------- Field / ESDF -----------------------

@dataclass
class CircleObs:
    x: float
    y: float
    r: float

    def contains(self, px, py):
        return math.hypot(px - self.x, py - self.y) <= self.r + 1e-9

class Field:
    """Simple field with circular obstacles. ESDF is min(distance_to_boundary) with sign negative inside."""
    def __init__(self):
        self.obstacles: List[CircleObs] = []

    def add(self, x, y, r):
        self.obstacles.append(CircleObs(x, y, max(5, r)))

    def remove_at(self, x, y):
        hit = None
        for c in self.obstacles:
            if c.contains(x, y):
                hit = c
                break
        if hit:
            self.obstacles.remove(hit)

    def esdf_at_xy(self, x: float, y: float) -> float:
        if not self.obstacles:
            # Distance to outside world is "infinite"; cap for visualization
            return 1e6
        dists = []
        for c in self.obstacles:
            d = math.hypot(x - c.x, y - c.y) - c.r
            dists.append(d)
        # ESDF is the minimum signed distance to any obstacle boundary
        return min(dists)

    def randomize(self, n=6):
        self.obstacles.clear()
        for _ in range(n):
            r = random.randint(18, 60)
            x = random.randint(r + 40, W - r - 40)
            y = random.randint(r + 40, H - r - 40)
            self.add(x, y, r)

# ----------------------- ORCA-style checker -----------------------

class OrcaChecker:
    def __init__(self, field: Field, radius: float = 12.0):
        self.field = field
        self.radius = radius
        self.points: List[Tuple[float, float]] = []  # debug breadcrumb

    def clearance(self, xy: Tuple[float, float], radius: Optional[float] = None) -> float:
        r = radius if radius is not None else self.radius
        return self.field.esdf_at_xy(xy[0], xy[1]) - r

    def is_orca_colliding(self,
        p1_xy: Tuple[float, float],
        p2_xy: Tuple[float, float],
        radius: Optional[float] = None,
        max_steps: int = 1000,
        eps: float = 1e-6,
        k: float = 4.0,
        visualize_cb=None
    ) -> bool:
        """
        True if the road from p1_xy to p2_xy with radius collides.
        Uses look-ahead steps (k) certified by the 1-Lipschitz property to avoid creeping.

        If visualize_cb is provided, it is called frequently with a dict payload to animate.
        """
        x1, y1 = p1_xy
        x2, y2 = p2_xy

        r = radius if radius is not None else self.radius

        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L == 0:
            coll = (self.field.esdf_at_xy(x1, y1) - r) < 0.0
            if visualize_cb:
                visualize_cb(dict(mode="done", collided=coll))
            return coll
        ux, uy = dx / L, dy / L

        # Early check
        if self.clearance(p1_xy, radius=r) <= 0.0:
            if visualize_cb:
                visualize_cb(dict(mode="done", collided=True))
            return True

        t = 0.0
        steps = 0
        while t < L - eps and steps < max_steps:
            steps += 1
            px, py = x1 + ux * t, y1 + uy * t
            d0 = self.clearance((px, py), radius=r)
            if d0 <= 0.0:
                if visualize_cb:
                    visualize_cb(dict(mode="done", collided=True))
                return True

            # Visualize local info
            if visualize_cb:
                visualize_cb(dict(mode="at_point", t=t, L=L, px=px, py=py, d0=d0))

            # If current clearance already exceeds remaining distance, we can certify the rest.
            if d0 >= (L - t):
                if visualize_cb:
                    visualize_cb(dict(mode="certify_tail", from_t=t, to_L=L, d0=d0))
                    visualize_cb(dict(mode="done", collided=False))
                return False

            # Propose a big step (k>1). We'll certify it before accepting.
            s = min(k * d0 + eps, L - t)

            # Certifying the step
            while True:
                p1x, p1y = px + ux * s, py + uy * s
                d1 = self.clearance((p1x, p1y), radius=r)

                if visualize_cb:
                    visualize_cb(dict(mode="try_step", t=t, s=s, px=px, py=py, p1x=p1x, p1y=p1y, d0=d0, d1=d1))

                # Lipschitz certificate: whole chunk [t, t+s] is safe
                if min(d0, d1) - s >= 0.0:
                    if visualize_cb:
                        visualize_cb(dict(mode="certified", from_t=t, to_t=t+s, px=px, py=py, p1x=p1x, p1y=p1y))
                    t += s
                    break

                # Found an unsafe endpoint -> collision somewhere in [t, t+s]
                if d1 <= 0.0:
                    self.points.append((p1x, p1y))
                    if visualize_cb:
                        visualize_cb(dict(mode="unsafe_endpoint", p1x=p1x, p1y=p1y, t=t, s=s))

                    lo, hi = t, t + s
                    for _ in range(40):  # logarithmic refinement
                        if hi - lo <= eps:
                            break
                        mid = 0.5 * (lo + hi)
                        mx, my = x1 + ux * mid, y1 + uy * mid
                        dm = self.clearance((mx, my), radius=r)
                        if visualize_cb:
                            visualize_cb(dict(mode="bisection", lo=lo, hi=hi, mid=mid, mx=mx, my=my, dm=dm))
                        if dm < 0.0:
                            hi = mid
                        else:
                            lo = mid
                    if visualize_cb:
                        visualize_cb(dict(mode="done", collided=True))
                    return True

                # Not certified safe, and not unsafe at the far end -> shrink step and retry
                s *= 0.5
                if s <= max(d0, eps):
                    # Fall back to the local safe step to keep making progress
                    if visualize_cb:
                        visualize_cb(dict(mode="fallback", add=t+max(d0, eps), from_t=t, d0=d0))
                    t += max(d0, eps)
                    break

        if visualize_cb:
            visualize_cb(dict(mode="done", collided=False))
        return False

# ----------------------- Visualization -----------------------

class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("ORCA-style Collision Checking Visualizer")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        self.field = Field()
        self.field.randomize(7)

        self.checker = OrcaChecker(self.field, radius=12.0)

        self.p1 = (120.0, H/2.0 - 120.0)
        self.p2 = (W-120.0, H/2.0 + 120.0)

        self.dragging = None  # 'p1', 'p2', 'obs_new', ('obs_resize', idx)
        self.new_obs_tmp = None  # (x, y, r)

        self.k = 4.0
        self.auto = False
        self.last_result = None
        self.step_events = []  # buffer of visualization events (for slow-mo)
        self.speed = 0.01  # events per frame; lower = slower, higher = faster

        # Persistent overlays (so visuals linger across frames)
        self.linger_frames = 36  # ~0.4s at 90 FPS; adjustable
        self.trail_circles = []  # items: (x, y, r, life)
        self.trail_segments = [] # items: (x0,y0,x1,y1, life, kind)  kind: 'proposed'|'certified'
        self.trail_points = []   # items: (x, y, life, kind) kind: 'mid'|'unsafe'

        self.reset_run()

    def reset_run(self):
        self.step_events.clear()
        self.last_result = None

    # ------------- Visualization callbacks -------------

    def viz_cb(self, payload):
        # Push events with a small pacing to see them; adjust slow_factor for slower animation
        self.step_events.append(payload)

    # ------------- Drawing -------------

    def draw_grid(self):
        self.screen.fill(BG)
        # light grid
        step = 25
        for x in range(0, W, step):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, H), 1)
        for y in range(0, H, step):
            pygame.draw.line(self.screen, GRID, (0, y), (W, y), 1)

    def draw_obstacles(self):
        for c in self.field.obstacles:
            pygame.draw.circle(self.screen, (60, 65, 75), (int(c.x), int(c.y)), int(c.r))
            pygame.draw.circle(self.screen, (95, 105, 120), (int(c.x), int(c.y)), int(c.r), 2)

    def draw_points_and_path(self):
        pygame.draw.circle(self.screen, GREEN, (int(self.p1[0]), int(self.p1[1])), 7)
        pygame.draw.circle(self.screen, BLUE, (int(self.p2[0]), int(self.p2[1])), 7)
        pygame.draw.line(self.screen, GREY, self.p1, self.p2, 2)

        # robot radius
        r = int(self.checker.radius)
        if r > 0:
            pygame.draw.circle(self.screen, (40, 160, 120), (int(self.p1[0]), int(self.p1[1])), r, 1)
            pygame.draw.circle(self.screen, (40, 120, 180), (int(self.p2[0]), int(self.p2[1])), r, 1)

    def draw_text(self, txt, x, y, color=UI, size=16):
        font = pygame.font.SysFont("consolas", size)
        s = font.render(txt, True, color)
        self.screen.blit(s, (x, y))


    def draw_trails(self):
        # Draw lingering circles
        keep_circles = []
        for (x, y, r, life) in self.trail_circles:
            if r > 0:
                pygame.draw.circle(self.screen, CYAN, (int(x), int(y)), int(r), 1)
            if life > 1: keep_circles.append((x, y, r, life - 1))
        self.trail_circles = keep_circles

        # Draw lingering segments
        keep_segs = []
        for (x0,y0,x1,y1, life, kind) in self.trail_segments:
            col = YELLOW if kind == 'proposed' else (120, 220, 130)
            width = 2 if kind == 'proposed' else 4
            pygame.draw.line(self.screen, col, (x0,y0), (x1,y1), width)
            if life > 1: keep_segs.append((x0,y0,x1,y1, life - 1, kind))
        self.trail_segments = keep_segs

        # Draw lingering points
        keep_pts = []
        for (x, y, life, kind) in self.trail_points:
            col = GREEN if kind == 'mid' else RED
            radius = 3 if kind == 'mid' else 6
            pygame.draw.circle(self.screen, col, (int(x), int(y)), radius)
            if life > 1: keep_pts.append((x, y, life - 1, kind))
        self.trail_points = keep_pts

    def draw_events(self):
        # Smooth pacing with fractional events-per-frame speed
        if not hasattr(self, "_event_accum"):
            self._event_accum = 0.0
        self._event_accum += max(0.05, float(self.speed))  # clamp tiny positives
        consumed = 0
        while self._event_accum >= 1.0 and self.step_events:
            evt = self.step_events.pop(0)
            self.render_event(evt)
            self._event_accum -= 1.0
            consumed += 1
        # If there's nothing to draw, slowly drain to avoid runaway accumulation
        if not self.step_events:
            self._event_accum = 0.0

    def render_event(self, e):
        mode = e.get("mode")
        if mode == "at_point":
            px, py = e["px"], e["py"]
            d0 = e["d0"]
            if d0 > 0:
                self.trail_circles.append((px, py, float(d0), int(self.linger_frames)))
            # mark current point lightly via a tiny lingering circle
            self.trail_points.append((px, py, int(self.linger_frames//2), 'mid'))
        elif mode == "try_step":
            px, py = e["px"], e["py"]
            p1x, p1y = e["p1x"], e["p1y"]
            self.trail_segments.append((px, py, p1x, p1y, int(self.linger_frames), 'proposed'))
        elif mode == "certified":
            self.trail_segments.append((e["px"], e["py"], e["p1x"], e["p1y"], int(self.linger_frames*2), 'certified'))  # certified lasts longer
        elif mode == "unsafe_endpoint":
            self.trail_points.append((e["p1x"], e["p1y"], int(self.linger_frames*2), 'unsafe'))
        elif mode == "bisection":
            mx, my = e["mx"], e["my"]
            dm = e["dm"]
            kind = 'mid' if dm >= 0 else 'unsafe'
            self.trail_points.append((mx, my, int(self.linger_frames), kind))
        elif mode == "fallback":
            pass  # implicit via subsequent "at_point"
        elif mode == "certify_tail":
            pass  # tail is the remainder; we keep it simple
        elif mode == "done":
            self.last_result = e.get("collided", None)

    # ------------- Interaction -------------

    def handle_input(self):
        mx, my = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_a:
                    self.auto = not self.auto
                    self.reset_run()
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.speed = min(30.0, self.speed * 1.25)
                if event.key == pygame.K_MINUS:
                    self.speed = max(0.0001, self.speed / 1.25)
                if event.key == pygame.K_0:
                    self.speed = 1.0
                if event.key == pygame.K_SPACE:
                    self.reset_run()
                    # run once (buffer events immediately, rendered over frames)
                    self.checker.is_orca_colliding(self.p1, self.p2, k=self.k, visualize_cb=self.viz_cb)
                if event.key == pygame.K_r:
                    self.field.randomize(7)
                    self.reset_run()
                if event.key == pygame.K_c:
                    self.field.obstacles.clear()
                    self.reset_run()
                if event.key == pygame.K_LEFTBRACKET:
                    self.k = max(1.0, self.k - 0.5)
                if event.key == pygame.K_RIGHTBRACKET:
                    self.k = min(12.0, self.k + 0.5)
                if event.key == pygame.K_s:
                    pygame.image.save(self.screen, "screenshot.png")
                if event.key == pygame.K_COMMA:
                    self.linger_frames = max(4, int(self.linger_frames * 0.8))
                if event.key == pygame.K_PERIOD:
                    self.linger_frames = min(240, int(self.linger_frames * 1.25))
                if event.key == pygame.K_SLASH:
                    self.linger_frames = 36
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left -> drag P1 or start new obstacle
                    if length((mx - self.p1[0], my - self.p1[1])) < 15:
                        self.dragging = 'p1'
                    else:
                        # start new obstacle
                        self.dragging = 'obs_new'
                        self.new_obs_tmp = (mx, my, 5)
                elif event.button == 3:  # right -> drag P2
                    if length((mx - self.p2[0], my - self.p2[1])) < 15:
                        self.dragging = 'p2'
                    else:
                        # remove obstacle if under mouse
                        self.field.remove_at(mx, my)
                        self.reset_run()
                elif event.button == 2:  # middle -> quick add obstacle
                    self.dragging = 'obs_new'
                    self.new_obs_tmp = (mx, my, 5)
                elif event.button == 4:  # wheel up -> increase radius
                    self.checker.radius = clamp(self.checker.radius + 1, 1, 80)
                    self.reset_run()
                elif event.button == 5:  # wheel down -> decrease radius
                    self.checker.radius = clamp(self.checker.radius - 1, 1, 80)
                    self.reset_run()
            if event.type == pygame.MOUSEBUTTONUP:
                if self.dragging == 'obs_new' and self.new_obs_tmp is not None:
                    x, y, r = self.new_obs_tmp
                    self.field.add(x, y, r)
                    self.new_obs_tmp = None
                self.dragging = None
            if event.type == pygame.MOUSEMOTION:
                if self.dragging == 'p1':
                    self.p1 = (mx, my)
                    self.reset_run()
                elif self.dragging == 'p2':
                    self.p2 = (mx, my)
                    self.reset_run()
                elif self.dragging == 'obs_new' and self.new_obs_tmp is not None:
                    x0, y0, _ = self.new_obs_tmp
                    r = max(5, int(math.hypot(mx - x0, my - y0)))
                    self.new_obs_tmp = (x0, y0, r)
                    self.reset_run()
        return True

    # ------------- Main loop -------------

    def run(self):
        while True:
            if not self.handle_input():
                break

            self.draw_grid()
            self.draw_obstacles()
            self.draw_points_and_path()

            # preview obstacle being created
            if self.new_obs_tmp is not None:
                x, y, r = self.new_obs_tmp
                pygame.draw.circle(self.screen, (70, 80, 95), (x, y), r, 2)

            # auto-run
            if self.auto and not self.step_events and self.last_result is None:
                self.checker.is_orca_colliding(self.p1, self.p2, k=self.k, visualize_cb=self.viz_cb)

            # draw lingering overlays then drain new events
            self.draw_trails()
            self.draw_events()

            # UI text
            info = [
                f"radius: {self.checker.radius:.1f}   k: {self.k:.1f}   auto: {'on' if self.auto else 'off'}",
                "SPACE: run once   A: auto   [ / ]: k- / k+   mouse wheel: radius",
                "LMB drag: P1   RMB drag: P2   mid/LMB: add obstacle (drag to size)   RMB: remove obstacle",
                f"speed: {self.speed:.2f} ev/frame   linger: {self.linger_frames}f   result: {self.last_result if self.last_result is not None else '—'}"
            ]
            y = 8
            for line in info:
                self.draw_text(line, 12, y)
                y += 18

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

# ----------------------- Entrypoint -----------------------

if __name__ == "__main__":
    App().run()