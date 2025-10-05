import numpy as np
from path_planer import PathPlaner



def RRT(path_obj: PathPlaner, start_xy, goal_xy,
        ROWS, COLS, STEP_SIZE=20, RRT_ITERATIONS=1000, GOAL_RADIUS=100):
    if start_xy is None or goal_xy is None:
        print("Start or goal not set")
        return None

    rrt = Tree()
    root_id = rrt.add_root(start_xy)

    # Early exit: goal already reachable from start
    if (np.hypot(goal_xy[0]-start_xy[0], goal_xy[1]-start_xy[1]) <= GOAL_RADIUS and
        not path_obj.is_orca_colliding(start_xy, goal_xy)):
        goal_id = rrt.add_child(pos=goal_xy, parent_id=root_id)
        path = rrt.path_from_id(goal_id)[::-1]  # start -> goal
        return path

    for _ in range(RRT_ITERATIONS):
        # 1) Sample
        random_point = (np.random.uniform(0, ROWS), np.random.uniform(0, COLS))

        # 2) Nearest (Tree.nearest returns (id, pos))
        nearest_id, nearest_pos = rrt.nearest(random_point)

        # 3) Steer
        dx = random_point[0] - nearest_pos[0]
        dy = random_point[1] - nearest_pos[1]
        dist = np.hypot(dx, dy)
        if dist == 0:
            continue

        if dist > STEP_SIZE:
            scale = STEP_SIZE / dist
            new_pos = (nearest_pos[0] + dx * scale, nearest_pos[1] + dy * scale)
        else:
            new_pos = random_point

        # keep inside bounds (optional but helpful)
        new_pos = (float(np.clip(new_pos[0], 0, ROWS)),
                   float(np.clip(new_pos[1], 0, COLS)))

        # 4) Collision check from nearest_pos -> new_pos
        if path_obj.is_orca_colliding(nearest_pos, new_pos):
            continue

        # 5) Add to tree
        new_id = rrt.add_child(pos=new_pos, parent_id=nearest_id)

        # 6) Goal check + connect straight to goal if clear
        to_goal = np.hypot(goal_xy[0] - new_pos[0], goal_xy[1] - new_pos[1])
        if to_goal <= GOAL_RADIUS and not path_obj.is_orca_colliding(new_pos, goal_xy):
            goal_id = rrt.add_child(pos=goal_xy, parent_id=new_id)
            path = rrt.path_from_id(goal_id)  # goal -> start 
            path.reverse()
            path_obj.points.extend(rrt.all_positions())
            return path

    # No path found within iteration budget
    return None







from path import Path
class Tree:
    """
    A simple parent-pointer tree for RRT:
      - nodes: id -> (row, col) float position
      - parent: id -> parent_id (or None for the root)
      - by_pos: rounded (row,col) -> id (to avoid brittle float equality)
    """
    def __init__(self, decimals=6):
        self.nodes = {}      # node_id -> (r, c)
        self.parent = {}     # node_id -> parent_id or None
        self.by_pos = {}     # (rounded_r, rounded_c) -> node_id
        self._next_id = 0
        self._decimals = decimals

    def _key(self, p):
        return (round(p[0], self._decimals), round(p[1], self._decimals))

    def clear(self):
        self.nodes.clear()
        self.parent.clear()
        self.by_pos.clear()
        self._next_id = 0

    # --- add/get ---
    def add_root(self, pos):
        nid = self._next_id; self._next_id += 1
        self.nodes[nid] = pos
        self.parent[nid] = None
        self.by_pos[self._key(pos)] = nid
        return nid
    
    def add_child(self, pos, parent_id):
        k = self._key(pos)
        if k in self.by_pos:
            nid = self.by_pos[k]
            # don't clobber existing parents unless missing
            if self.parent.get(nid) is None and parent_id is not None:
                self.parent[nid] = parent_id
            return nid
        nid = self._next_id; self._next_id += 1
        self.nodes[nid] = pos
        self.parent[nid] = parent_id
        self.by_pos[k] = nid
        return nid

    def id_of_pos(self, pos):
        return self.by_pos.get(self._key(pos), None)

    def pos_of(self, node_id):
        return self.nodes[node_id]

    # --- queries ---
    def nearest(self, pos):
        """Return (nearest_id, nearest_pos) with linear scan."""
        best_id, best_d2 = None, float('inf')
        pr, pc = pos
        for nid, (r, c) in self.nodes.items():
            d2 = (r - pr) * (r - pr) + (c - pc) * (c - pc)
            if d2 < best_d2:
                best_d2, best_id = d2, nid
        return best_id, self.nodes[best_id]

    def path_from_id(self, node_id):
        """Return path [node, ..., root] as positions (child->root)."""
        path = Path([])
        nid = node_id
        while nid is not None:
            path.add_point(self.nodes[nid])
            nid = self.parent[nid]
        return path
    
    def get_parent(self, node_id):
        return self.parent[node_id]

    def path_from_pos(self, pos):
        nid = self.id_of_pos(pos)
        return None if nid is None else self.path_from_id(nid)

    # --- iteration helpers for drawing ---
    def all_positions(self):
        return list(self.nodes.values())

    def edges(self):
        """Yield (parent_pos, child_pos) tuples."""
        for nid, p in self.nodes.items():
            pid = self.parent[nid]
            if pid is not None:
                yield (self.nodes[pid], p)

    def print_tree(self, *, decimals=3, show_ids=True, max_nodes=None, stream=None):
        """
        Pretty-print the tree as ASCII.

        Args:
            decimals (int): number of decimals for positions.
            show_ids (bool): include node IDs in labels.
            max_nodes (int|None): stop after printing this many nodes (shows ellipsis).
            stream: file-like to write to (defaults to sys.stdout).
        """
        import sys
        out = stream or sys.stdout

        if not self.nodes:
            out.write("(empty tree)\n")
            return

        # Build children adjacency
        children = {nid: [] for nid in self.nodes}
        roots = []
        for nid, pid in self.parent.items():
            if pid is None:
                roots.append(nid)
            else:
                children.setdefault(pid, []).append(nid)

        # Deterministic order
        for ch in children.values():
            ch.sort()
        roots.sort()

        def label(nid):
            r, c = self.nodes[nid]
            pos = f"({r:.{decimals}f}, {c:.{decimals}f})"
            return f"[{nid}] {pos}" if show_ids else pos

        printed = 0
        stop = False

        def recurse(nid, prefix="", is_last=True):
            nonlocal printed, stop
            if stop:
                return
            connector = "└─" if is_last else "├─"
            if prefix == "":  # root line
                out.write(f"{label(nid)}\n")
            else:
                out.write(f"{prefix}{connector}{label(nid)}\n")
            printed += 1
            if max_nodes is not None and printed >= max_nodes:
                stop = True
                return
            ch = children.get(nid, [])
            new_prefix = prefix + ("  " if is_last else "│ ")
            for i, cid in enumerate(ch):
                recurse(cid, new_prefix, i == len(ch) - 1)
                if stop:
                    return

        # Print all roots (there's usually just one)
        for i, rid in enumerate(roots):
            recurse(rid, "", True)
            if stop:
                out.write("… (truncated)\n")
                break
            if i < len(roots) - 1:
                out.write("\n")
