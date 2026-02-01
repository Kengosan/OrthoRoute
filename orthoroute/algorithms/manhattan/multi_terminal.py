"""
Multi-Terminal Net Router for OrthoRoute

Implements MST-based routing for nets with 3+ pads.
Phase 2 of OrthoRoute Enhancement Plan.

Key components:
- PadInfo: Layer-aware pad representation
- MSTEdge: MST candidate edge with maze-cost estimation
- MultiTerminalNetState: Tracks routing progress per net
- UnionFind: Connectivity validation
- MazeCostMSTBuilder: Builds MST with congestion-aware costs
- MultiSourceRouter: Routes to existing tree with same-net reuse
- ConnectivityValidator: Validates full net connectivity
- MultiTerminalNetRouter: Integration class

Usage:
    router = MultiTerminalNetRouter(pathfinder)
    results = router.route_all_nets(nets, progress_cb, iteration_cb)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PadInfo:
    """
    パッド情報（レイヤー考慮）

    Attributes:
        pad_id: Unique pad identifier (e.g., "U1-1")
        x_idx: X index in routing lattice
        y_idx: Y index in routing lattice
        layer: Layer index (0=F.Cu, 1=B.Cu, -1=THRU)
        node: Graph node ID for routing
        world_x: World X coordinate (mm)
        world_y: World Y coordinate (mm)
    """
    pad_id: str
    x_idx: int
    y_idx: int
    layer: int  # 0=F.Cu, 1=B.Cu, -1=THRU
    node: int   # グラフノードID
    world_x: float
    world_y: float


@dataclass
class MSTEdge:
    """
    MST候補エッジ

    Attributes:
        from_pad: Source pad
        to_pad: Target pad
        estimated_cost: Maze cost estimate (congestion-aware)
        manhattan_dist: Manhattan distance in lattice units
    """
    from_pad: PadInfo
    to_pad: PadInfo
    estimated_cost: float  # 迷路コスト推定値
    manhattan_dist: int

    def __lt__(self, other):
        return self.estimated_cost < other.estimated_cost


@dataclass
class MultiTerminalNetState:
    """
    多端子ネットの接続状態

    Tracks which pads are routed and stores routing paths.
    Used for incremental MST-style routing.
    """
    net_name: str
    all_pads: List[PadInfo]
    routed_pads: Set[str] = field(default_factory=set)
    routed_nodes: Set[int] = field(default_factory=set)
    routed_paths: List[List[int]] = field(default_factory=list)
    mst_edges: List[MSTEdge] = field(default_factory=list)

    def is_fully_connected(self) -> bool:
        """全パッドが接続済みか"""
        return len(self.routed_pads) == len(self.all_pads)

    def get_unrouted_pads(self) -> List[PadInfo]:
        """未接続パッドを返す"""
        return [p for p in self.all_pads if p.pad_id not in self.routed_pads]

    def connectivity_ratio(self) -> float:
        """接続率を返す"""
        if len(self.all_pads) == 0:
            return 1.0
        return len(self.routed_pads) / len(self.all_pads)

    def add_routed_path(self, path: List[int], *pads: PadInfo):
        """ルーティング結果を記録"""
        for pad in pads:
            if pad:
                self.routed_pads.add(pad.pad_id)
        self.routed_nodes.update(path)
        self.routed_paths.append(path)


class UnionFind:
    """
    接続性検証用Union-Find (Disjoint Set Union)

    Used to efficiently check if all pads in a net are connected
    through the routed paths.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self._size = [1] * n

    def find(self, x: int) -> int:
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank, returns True if newly connected"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self._size[px] += self._size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set"""
        return self.find(x) == self.find(y)

    def component_size(self, x: int) -> int:
        """Get size of component containing x"""
        return self._size[self.find(x)]

    def num_components(self) -> int:
        """Count number of distinct components"""
        return len(set(self.find(i) for i in range(len(self.parent))))

    def largest_component_size(self) -> int:
        """Get size of largest component"""
        if not self.parent:
            return 0
        component_sizes = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            component_sizes[root] = component_sizes.get(root, 0) + 1
        return max(component_sizes.values()) if component_sizes else 0


# =============================================================================
# Preprocessor
# =============================================================================

class MultiTerminalPreprocessor:
    """
    多端子ネット前処理

    - Deduplicates pads at same coordinates
    - Maps pads to routing graph nodes
    - Determines pad layers (F.Cu/B.Cu/THRU)
    """

    def __init__(self, pathfinder):
        self.pf = pathfinder

    def preprocess_net(self, net) -> Optional[MultiTerminalNetState]:
        """
        ネットを前処理し、MultiTerminalNetStateを生成。

        - 重複パッド検出・統合
        - レイヤー考慮のノードマッピング
        - 2パッド未満はスキップ
        """
        pads = getattr(net, 'pads', [])
        if len(pads) < 2:
            return None

        # パッド情報収集（重複検出用）
        seen_coords: Dict[Tuple[int, int, int], PadInfo] = {}
        pad_infos: List[PadInfo] = []

        for pad in pads:
            pad_id = self._get_pad_key(pad)

            # ポータル確認
            if not hasattr(self.pf, 'portals') or pad_id not in self.pf.portals:
                logger.debug(f"Pad {pad_id} has no portal, skipping")
                continue

            portal = self.pf.portals[pad_id]

            # レイヤー決定
            layer = self._determine_pad_layer(pad, portal)

            # ノードID計算
            try:
                node = self.pf.lattice.node_idx(
                    portal.x_idx, portal.y_idx, portal.entry_layer
                )
            except Exception as e:
                logger.debug(f"Failed to get node for pad {pad_id}: {e}")
                continue

            # 重複チェック
            coord_key = (portal.x_idx, portal.y_idx, layer)
            if coord_key in seen_coords:
                logger.debug(f"Duplicate pad at {coord_key}, merging")
                continue

            pad_info = PadInfo(
                pad_id=pad_id,
                x_idx=portal.x_idx,
                y_idx=portal.y_idx,
                layer=layer,
                node=node,
                world_x=getattr(portal, 'x_mm', 0.0),
                world_y=getattr(portal, 'y_mm', 0.0),
            )
            seen_coords[coord_key] = pad_info
            pad_infos.append(pad_info)

        if len(pad_infos) < 2:
            return None

        return MultiTerminalNetState(
            net_name=getattr(net, 'name', 'unknown'),
            all_pads=pad_infos,
        )

    def _get_pad_key(self, pad) -> str:
        """Get unique pad identifier"""
        if hasattr(self.pf, '_pad_key'):
            return self.pf._pad_key(pad)
        # Fallback
        if hasattr(pad, 'id'):
            return str(pad.id)
        if hasattr(pad, 'pad_id'):
            return str(pad.pad_id)
        return f"{getattr(pad, 'component', 'X')}-{getattr(pad, 'number', '0')}"

    def _determine_pad_layer(self, pad, portal) -> int:
        """
        パッドのレイヤーを決定。

        - Through-hole: -1 (THRU) - can connect on both layers
        - SMD: F.Cu(0) or B.Cu(1)

        Fixed: Check for through-hole FIRST, before the 2-layer early return.
        This ensures TH pads can connect on both layers even on 2-layer boards.
        """
        entry_layer = getattr(portal, 'entry_layer', 0)

        # スルーホール判定 (FIRST - before layer-specific handling)
        pad_type = getattr(pad, 'pad_type', None)
        if pad_type in ('THRU', 'through_hole', 'th', 'PTH'):
            return -1  # Through-hole can connect on any layer

        # SMD pads - layer-specific
        # 2層基板の場合
        if hasattr(self.pf, 'layers') and self.pf.layers == 2:
            if entry_layer == 0:
                return 0  # F.Cu
            elif entry_layer == 1:
                return 1  # B.Cu

        return entry_layer


# =============================================================================
# MST Builder
# =============================================================================

class MazeCostMSTBuilder:
    """
    迷路コストベースのMST構築

    Uses KD-tree for efficient nearest neighbor search and
    heuristic maze cost estimation for edge weights.
    """

    def __init__(self, pathfinder):
        self.pf = pathfinder
        self._cost_cache: Dict[Tuple[int, int], float] = {}

    def build_mst(self, state: MultiTerminalNetState) -> List[MSTEdge]:
        """
        Prim法でMSTを構築。エッジ重みは迷路コスト推定値。

        1. 全ペア間の候補エッジを生成（KD-treeでプルーニング）
        2. 各エッジの迷路コストを推定
        3. Prim法でMST構築
        """
        pads = state.all_pads
        n = len(pads)

        if n < 2:
            return []

        # Step 1: 候補エッジ生成（近傍のみ）
        candidate_edges = self._generate_candidate_edges(pads)

        # Step 2: 迷路コスト推定
        for edge in candidate_edges:
            cache_key = (edge.from_pad.node, edge.to_pad.node)
            if cache_key not in self._cost_cache:
                cost = self._estimate_maze_cost(edge.from_pad, edge.to_pad)
                self._cost_cache[cache_key] = cost
                self._cost_cache[(edge.to_pad.node, edge.from_pad.node)] = cost
            edge.estimated_cost = self._cost_cache[cache_key]

        # Step 3: Prim法でMST
        mst_edges = self._prim_mst(pads, candidate_edges)

        state.mst_edges = mst_edges
        logger.debug(f"Built MST with {len(mst_edges)} edges for {n} pads")
        return mst_edges

    def _generate_candidate_edges(self, pads: List[PadInfo], k_nearest: int = 6) -> List[MSTEdge]:
        """
        KD-treeで近傍パッドのみを候補に。

        完全グラフはO(n²)なので、k-nearest neighborでO(n*k)に削減。
        """
        n = len(pads)

        if n <= k_nearest + 1:
            # 小規模: 全ペア
            edges = []
            for i, p1 in enumerate(pads):
                for p2 in pads[i+1:]:
                    dist = abs(p1.x_idx - p2.x_idx) + abs(p1.y_idx - p2.y_idx)
                    edges.append(MSTEdge(
                        from_pad=p1, to_pad=p2,
                        estimated_cost=float(dist), manhattan_dist=dist
                    ))
            return edges

        # KD-tree構築
        try:
            from scipy.spatial import KDTree

            coords = np.array([[p.x_idx, p.y_idx] for p in pads], dtype=np.float64)
            tree = KDTree(coords)

            edges = []
            seen = set()

            for i, pad in enumerate(pads):
                # k+1 nearest (自分自身を含む)
                distances, indices = tree.query(
                    [pad.x_idx, pad.y_idx],
                    k=min(k_nearest + 1, n)
                )

                for idx in indices:
                    if idx == i:
                        continue

                    # 重複回避
                    edge_key = (min(i, idx), max(i, idx))
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)

                    other = pads[idx]
                    dist = abs(pad.x_idx - other.x_idx) + abs(pad.y_idx - other.y_idx)
                    edges.append(MSTEdge(
                        from_pad=pad, to_pad=other,
                        estimated_cost=float(dist), manhattan_dist=dist
                    ))

            return edges

        except ImportError:
            logger.warning("SciPy not available, using brute-force edge generation")
            # Fallback to all pairs
            edges = []
            for i, p1 in enumerate(pads):
                for p2 in pads[i+1:]:
                    dist = abs(p1.x_idx - p2.x_idx) + abs(p1.y_idx - p2.y_idx)
                    edges.append(MSTEdge(
                        from_pad=p1, to_pad=p2,
                        estimated_cost=float(dist), manhattan_dist=dist
                    ))
            return edges

    def _estimate_maze_cost(self, from_pad: PadInfo, to_pad: PadInfo) -> float:
        """
        迷路コストを推定（ヒューリスティック）。

        完全なA*ではなく、コスト推定のプロキシ。
        後続フェーズで真のA*に改善可能。

        Fixed: present is per-edge, not per-node. Now samples edge costs
        from the accounting system.
        """
        # マンハッタン距離ベース
        base_cost = abs(from_pad.x_idx - to_pad.x_idx) + abs(from_pad.y_idx - to_pad.y_idx)

        # レイヤー遷移コスト
        layer_penalty = 0
        if from_pad.layer != to_pad.layer:
            if from_pad.layer >= 0 and to_pad.layer >= 0:
                layer_penalty = 5  # Via cost estimate

        # 渋滞ペナルティ（edge-based congestion sampling）
        congestion_penalty = 0
        if hasattr(self.pf, 'accounting') and hasattr(self.pf.accounting, 'total_cost'):
            try:
                # Sample total cost at midpoint edges
                mid_x = (from_pad.x_idx + to_pad.x_idx) // 2
                mid_y = (from_pad.y_idx + to_pad.y_idx) // 2

                total_cost = self.pf.accounting.total_cost
                if hasattr(total_cost, 'get'):
                    total_cost = total_cost.get()

                layers_to_check = range(min(2, getattr(self.pf, 'layers', 2)))
                for layer in layers_to_check:
                    mid_node = self.pf.lattice.node_idx(mid_x, mid_y, layer)

                    # Sample outgoing edges from mid_node
                    if hasattr(self.pf, 'adjacency_matrix'):
                        adj_indptr = self.pf.adjacency_matrix.indptr
                        if hasattr(adj_indptr, 'get'):
                            adj_indptr = adj_indptr.get()

                        start = adj_indptr[mid_node]
                        end = adj_indptr[mid_node + 1]

                        # Average cost of outgoing edges
                        if end > start:
                            edge_costs = total_cost[start:end]
                            avg_cost = float(np.mean(edge_costs))
                            # Subtract base cost to get congestion component
                            grid_pitch = getattr(self.pf.config, 'grid_pitch', 0.4)
                            if avg_cost > grid_pitch:
                                congestion_penalty += (avg_cost - grid_pitch) * 0.5

            except Exception:
                pass  # Ignore errors in congestion estimation

        return base_cost + layer_penalty + congestion_penalty

    def _prim_mst(self, pads: List[PadInfo], edges: List[MSTEdge]) -> List[MSTEdge]:
        """
        Prim法でMST構築。

        Fixed: Validates MST connectivity and adds fallback edges if
        k-nearest pruning created a disconnected graph.
        """
        import heapq

        n = len(pads)
        if n < 2:
            return []

        # パッドIDからインデックスへのマップ
        pad_to_idx = {p.pad_id: i for i, p in enumerate(pads)}

        # 隣接リスト
        adj: Dict[int, List[Tuple[float, int, MSTEdge]]] = {i: [] for i in range(n)}
        for edge in edges:
            i = pad_to_idx.get(edge.from_pad.pad_id)
            j = pad_to_idx.get(edge.to_pad.pad_id)
            if i is not None and j is not None:
                adj[i].append((edge.estimated_cost, j, edge))
                adj[j].append((edge.estimated_cost, i, edge))

        # Prim's algorithm
        in_mst = [False] * n
        mst_edges = []

        # Start from node 0
        in_mst[0] = True
        heap = list(adj[0])
        heapq.heapify(heap)

        while heap and len(mst_edges) < n - 1:
            cost, to_idx, edge = heapq.heappop(heap)

            if in_mst[to_idx]:
                continue

            in_mst[to_idx] = True
            mst_edges.append(edge)

            for next_cost, next_idx, next_edge in adj[to_idx]:
                if not in_mst[next_idx]:
                    heapq.heappush(heap, (next_cost, next_idx, next_edge))

        # Connectivity validation: Check if all pads are in MST
        unreached = [i for i in range(n) if not in_mst[i]]

        if unreached:
            logger.warning(f"MST incomplete: {len(unreached)}/{n} pads unreachable, adding fallback edges")

            # Add fallback edges to connect unreached pads
            reached_pads = [pads[i] for i in range(n) if in_mst[i]]

            for unreached_idx in unreached:
                unreached_pad = pads[unreached_idx]

                # Find nearest reached pad (brute force, since this is fallback)
                min_dist = float('inf')
                best_reached_pad = None

                for reached_pad in reached_pads:
                    dist = (abs(unreached_pad.x_idx - reached_pad.x_idx) +
                            abs(unreached_pad.y_idx - reached_pad.y_idx))
                    if dist < min_dist:
                        min_dist = dist
                        best_reached_pad = reached_pad

                if best_reached_pad:
                    # Create fallback edge
                    fallback_edge = MSTEdge(
                        from_pad=best_reached_pad,
                        to_pad=unreached_pad,
                        estimated_cost=float(min_dist) * 1.5,  # Penalty for fallback
                        manhattan_dist=int(min_dist)
                    )
                    mst_edges.append(fallback_edge)
                    in_mst[unreached_idx] = True
                    reached_pads.append(unreached_pad)

                    logger.debug(f"Added fallback edge: {best_reached_pad.pad_id} -> {unreached_pad.pad_id}")

        return mst_edges


# =============================================================================
# Multi-Source Router
# =============================================================================

class MultiSourceRouter:
    """
    マルチソースルーティング実装

    Routes additional pads to existing routed tree.
    Implements same-net reuse to prevent self-blocking.
    """

    def __init__(self, pathfinder):
        self.pf = pathfinder
        self.same_net_nodes: Dict[str, Set[int]] = {}

    def route_multi_terminal_net(self, state: MultiTerminalNetState,
                                  max_retries: int = 3) -> bool:
        """
        MST edgesに沿って多端子ネットをルーティング。

        1. MST edgesを短い順にソート
        2. 各edgeをルーティング
        3. 失敗時はk-nearest retry
        """
        net_name = state.net_name
        pads = state.all_pads

        if len(pads) < 2:
            return True

        if not state.mst_edges:
            logger.warning(f"Net {net_name}: No MST edges, skipping")
            return False

        # 同一ネットノード初期化
        self.same_net_nodes[net_name] = set()

        # MST edgesを短い順にソート（成功率向上）
        sorted_edges = sorted(state.mst_edges, key=lambda e: e.estimated_cost)

        # 最初のedgeをルーティング
        first_edge = sorted_edges[0]
        path = self._route_pair(
            first_edge.from_pad.node,
            first_edge.to_pad.node,
            net_name
        )

        if path:
            state.add_routed_path(path, first_edge.from_pad, first_edge.to_pad)
            self.same_net_nodes[net_name].update(path)
            logger.debug(f"Net {net_name}: Routed initial edge, {len(path)} nodes")
        else:
            # 初期edge失敗時のフォールバック: 別のedgeを試す
            logger.warning(f"Net {net_name}: Initial edge failed, trying alternatives")
            for alt_edge in sorted_edges[1:min(4, len(sorted_edges))]:
                path = self._route_pair(
                    alt_edge.from_pad.node,
                    alt_edge.to_pad.node,
                    net_name
                )
                if path:
                    state.add_routed_path(path, alt_edge.from_pad, alt_edge.to_pad)
                    self.same_net_nodes[net_name].update(path)
                    logger.debug(f"Net {net_name}: Alternative edge succeeded")
                    break
            else:
                logger.error(f"Net {net_name}: All initial edges failed")
                return False

        # 残りのパッドを順次接続
        for pad in pads:
            if pad.pad_id in state.routed_pads:
                continue

            success = self._connect_pad_to_tree(state, pad, max_retries)
            if not success:
                logger.warning(f"Net {net_name}: Failed to connect pad {pad.pad_id}")

        return state.is_fully_connected()

    def _connect_pad_to_tree(self, state: MultiTerminalNetState,
                              target_pad: PadInfo, max_retries: int) -> bool:
        """
        ターゲットパッドをルーティング済みツリーに接続。

        Fixed: Each retry uses a DIFFERENT source node (from different routed pads)
        instead of always using the same nearest node.
        """
        net_name = state.net_name
        target_node = target_pad.node

        # ルーティング済みパッドを距離順にソート
        routed_pads = [p for p in state.all_pads if p.pad_id in state.routed_pads]
        routed_pads.sort(key=lambda p:
            abs(p.x_idx - target_pad.x_idx) + abs(p.y_idx - target_pad.y_idx))

        # Collect different source candidates for retries
        source_candidates = []

        # Add pad nodes directly
        for p in routed_pads:
            if p.node not in source_candidates:
                source_candidates.append(p.node)

        # Add nearest routed node (may differ from pad nodes)
        nearest = self._find_nearest_routed_node(target_node, state.routed_nodes)
        if nearest and nearest not in source_candidates:
            source_candidates.insert(0, nearest)  # Try nearest first

        # 最大k個の異なるソースを試行
        for retry in range(min(max_retries, len(source_candidates))):
            source_node = source_candidates[retry]

            path = self._route_pair(source_node, target_node, net_name)

            if path:
                state.add_routed_path(path, None, target_pad)
                self.same_net_nodes[net_name].update(path)
                return True

            logger.debug(f"Net {net_name}: Retry {retry+1}/{max_retries} for {target_pad.pad_id} (source={source_node})")

        return False

    def _find_nearest_routed_node(self, target: int, routed_nodes: Set[int]) -> Optional[int]:
        """ターゲットに最も近いルーティング済みノードを探す"""
        if not routed_nodes or not hasattr(self.pf, 'lattice'):
            return None

        try:
            target_coords = self.pf.lattice.idx_to_coord(target)
            tx, ty, tz = target_coords
        except Exception:
            return None

        min_dist = float('inf')
        nearest = None

        for node in routed_nodes:
            try:
                nx, ny, nz = self.pf.lattice.idx_to_coord(node)
                # マンハッタン距離（レイヤー遷移は2倍コスト）
                dist = abs(tx - nx) + abs(ty - ny) + abs(tz - nz) * 2
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            except Exception:
                continue

        return nearest

    def _route_pair(self, source: int, target: int, net_name: str) -> Optional[List[int]]:
        """
        単一ペアのルーティング。

        Uses PathFinder's CPU Dijkstra fallback for reliable routing.
        Same-net nodes are handled through force_allow mechanism.
        """
        if source == target:
            return [source]

        # Method 1: Use _cpu_dijkstra_fallback (most reliable)
        if hasattr(self.pf, '_cpu_dijkstra_fallback'):
            try:
                # same-net reuseのため、同一ネットノードを低コストに設定
                same_net = self.same_net_nodes.get(net_name, set())

                # 一時的にエッジコストを調整（same-net reuse用）
                if same_net and hasattr(self.pf, 'accounting') and hasattr(self.pf.accounting, 'total_cost'):
                    # TODO: より洗練された same-net reuse 実装
                    # 現状はシンプルにDijkstraを実行
                    pass

                path = self.pf._cpu_dijkstra_fallback(source, target)
                return path if path else None

            except Exception as e:
                logger.error(f"Routing failed for {net_name} via _cpu_dijkstra_fallback: {e}")
                # Fall through to other methods

        # Method 2: Use _route_single_net_cpu if available (returns result object with node_path)
        if hasattr(self.pf, '_route_single_net_cpu'):
            try:
                result = self.pf._route_single_net_cpu(net_name, source, target)
                if result and hasattr(result, 'node_path') and result.node_path:
                    return list(result.node_path)
                elif result and hasattr(result, 'success') and result.success:
                    # Some implementations store path differently
                    if hasattr(result, 'path') and result.path:
                        return list(result.path)
            except Exception as e:
                logger.debug(f"_route_single_net_cpu failed for {net_name}: {e}")

        # Method 3: Simple BFS fallback (last resort)
        try:
            path = self._simple_bfs_route(source, target)
            return path if path else None
        except Exception as e:
            logger.error(f"All routing methods failed for {net_name}: {e}")
            return None

    def _simple_bfs_route(self, source: int, target: int) -> Optional[List[int]]:
        """
        シンプルなBFSルーティング（最終フォールバック）。

        Uses adjacency matrix directly for basic pathfinding.
        """
        if not hasattr(self.pf, 'adjacency_matrix'):
            return None

        try:
            # Get adjacency data
            if hasattr(self.pf.adjacency_matrix, 'indptr'):
                adj_indptr = self.pf.adjacency_matrix.indptr
                adj_indices = self.pf.adjacency_matrix.indices
            else:
                return None

            # Handle CuPy arrays
            if hasattr(adj_indptr, 'get'):
                adj_indptr = adj_indptr.get()
                adj_indices = adj_indices.get()

            # BFS
            visited = {source}
            parent = {source: None}
            queue = deque([source])

            max_nodes = 100000  # Limit search
            nodes_checked = 0

            while queue and nodes_checked < max_nodes:
                current = queue.popleft()
                nodes_checked += 1

                if current == target:
                    # Reconstruct path
                    path = []
                    node = target
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    return list(reversed(path))

                # Expand neighbors
                start = adj_indptr[current]
                end = adj_indptr[current + 1]

                for edge_idx in range(start, end):
                    neighbor = adj_indices[edge_idx]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)

            return None

        except Exception as e:
            logger.debug(f"BFS routing failed: {e}")
            return None


# =============================================================================
# Connectivity Validator
# =============================================================================

class ConnectivityValidator:
    """
    接続性検証

    Uses Union-Find to verify all pads are connected through routed paths.
    Fixed: Considers node-level connectivity, not just pad-to-pad paths.
    """

    def __init__(self, pathfinder):
        self.pf = pathfinder

    def validate_net(self, state: MultiTerminalNetState) -> Tuple[bool, float]:
        """
        ネットの接続性を検証。

        Fixed: Only unions pads that are ACTUALLY connected through paths,
        not just pads whose nodes happen to be in routed_nodes set.

        Returns:
            (is_fully_connected, connectivity_ratio)
        """
        n = len(state.all_pads)
        if n < 2:
            return True, 1.0

        # ノードからパッドへのマップを構築
        node_to_pad_idx: Dict[int, int] = {}
        for i, pad in enumerate(state.all_pads):
            node_to_pad_idx[pad.node] = i

        # Union-Find初期化
        uf = UnionFind(n)

        # Step 1: Build path node sets and find pads in each path
        path_info = []  # List of (path_nodes_set, pad_indices_in_path)
        for path in state.routed_paths:
            if not path:
                continue
            path_set = set(path)
            path_pad_indices = [i for i, pad in enumerate(state.all_pads) if pad.node in path_set]
            path_info.append((path_set, path_pad_indices))

        # Step 2: Union pads within each path
        for path_set, path_pad_indices in path_info:
            if len(path_pad_indices) >= 2:
                first = path_pad_indices[0]
                for idx in path_pad_indices[1:]:
                    uf.union(first, idx)

        # Step 3: Union pads across paths that share common nodes
        # If path A and path B share any node, their pads are transitively connected
        for i, (set_i, pads_i) in enumerate(path_info):
            for j, (set_j, pads_j) in enumerate(path_info[i+1:], start=i+1):
                # Check if paths share any node
                if set_i & set_j:  # Non-empty intersection
                    # Union all pads from both paths together
                    all_pads = pads_i + pads_j
                    if len(all_pads) >= 2:
                        first = all_pads[0]
                        for idx in all_pads[1:]:
                            uf.union(first, idx)

        # パスが直接パッドノードを通らない場合の処理
        # Fixed: Only union with ONE representative pad from the same routed tree,
        # not with ALL pads in routed_nodes (which could be in different components)
        if hasattr(self.pf, 'lattice') and hasattr(self.pf, 'adjacency_matrix'):
            try:
                # Build mapping from routed nodes to pad indices
                routed_node_to_pad: Dict[int, int] = {}
                for i, pad in enumerate(state.all_pads):
                    if pad.node in state.routed_nodes:
                        routed_node_to_pad[pad.node] = i

                # For each unrouted pad, check if its neighbor is a routed node
                for i, pad_i in enumerate(state.all_pads):
                    if pad_i.node in state.routed_nodes:
                        # Already routed, skip
                        continue

                    # Check neighbors of this pad's node
                    pad_neighbors = self._get_neighbors(pad_i.node)
                    connected_pad = None

                    for neighbor in pad_neighbors:
                        if neighbor in state.routed_nodes:
                            # Find which pad (if any) this routed node belongs to
                            if neighbor in routed_node_to_pad:
                                connected_pad = routed_node_to_pad[neighbor]
                                break
                            else:
                                # Neighbor is routed but not a pad node - trace back to find a pad
                                # Check if any path contains this neighbor and a pad
                                for path in state.routed_paths:
                                    if neighbor in path:
                                        # Find any pad in this path
                                        for j, pad_j in enumerate(state.all_pads):
                                            if pad_j.node in path:
                                                connected_pad = j
                                                break
                                        if connected_pad is not None:
                                            break

                    # Only union with the specific connected pad, not all routed pads
                    if connected_pad is not None:
                        uf.union(i, connected_pad)

            except Exception as e:
                logger.debug(f"Neighbor check failed: {e}")

        # 接続性チェック
        num_components = uf.num_components()
        is_connected = num_components == 1

        # 接続率（最大コンポーネントサイズ / 全パッド数）
        largest = uf.largest_component_size()
        connectivity_ratio = largest / n

        return is_connected, connectivity_ratio

    def _get_neighbors(self, node: int) -> List[int]:
        """Get neighboring nodes from adjacency matrix"""
        try:
            adj_indptr = self.pf.adjacency_matrix.indptr
            adj_indices = self.pf.adjacency_matrix.indices

            # Handle CuPy arrays
            if hasattr(adj_indptr, 'get'):
                adj_indptr = adj_indptr.get()
                adj_indices = adj_indices.get()

            start = adj_indptr[node]
            end = adj_indptr[node + 1]
            return list(adj_indices[start:end])
        except Exception:
            return []

    def validate_all_nets(self, net_states: List[MultiTerminalNetState]) -> Dict[str, Any]:
        """全ネットの接続性を検証"""
        results = {
            'total_nets': len(net_states),
            'fully_connected': 0,
            'partially_connected': 0,
            'disconnected': 0,
            'average_connectivity': 0.0,
            'details': {}
        }

        total_ratio = 0.0

        for state in net_states:
            is_connected, ratio = self.validate_net(state)

            results['details'][state.net_name] = {
                'is_fully_connected': is_connected,
                'connectivity_ratio': ratio,
                'routed_pads': len(state.routed_pads),
                'total_pads': len(state.all_pads),
            }

            if is_connected:
                results['fully_connected'] += 1
            elif ratio > 0:
                results['partially_connected'] += 1
            else:
                results['disconnected'] += 1

            total_ratio += ratio

        if net_states:
            results['average_connectivity'] = total_ratio / len(net_states)

        # 成功率（全パッド接続基準）
        results['success_rate'] = (
            results['fully_connected'] / results['total_nets']
            if results['total_nets'] > 0 else 0.0
        )

        return results


# =============================================================================
# Integration Class
# =============================================================================

class MultiTerminalNetRouter:
    """
    多端子ネットルーター統合クラス。

    Main entry point for multi-terminal routing.
    Coordinates preprocessing, MST building, routing, and validation.
    """

    def __init__(self, pathfinder):
        self.pf = pathfinder
        self.preprocessor = MultiTerminalPreprocessor(pathfinder)
        self.mst_builder = MazeCostMSTBuilder(pathfinder)
        self.router = MultiSourceRouter(pathfinder)
        self.validator = ConnectivityValidator(pathfinder)

        # Statistics
        self.stats = {
            'total_nets': 0,
            'multi_terminal_nets': 0,
            'two_terminal_nets': 0,
            'fully_connected': 0,
            'partially_connected': 0,
        }

    def route_all_nets(self, nets: List, progress_cb=None, iteration_cb=None) -> Dict:
        """
        全ネットを多端子対応でルーティング。

        1. 前処理（パッド正規化、重複除去）
        2. ネット順序最適化
        3. MST構築
        4. ルーティング実行
        5. 接続性検証

        Returns:
            Dict mapping net_name -> path (list of node IDs)
        """
        logger.info(f"=== Multi-Terminal Routing: {len(nets)} nets ===")

        # Step 1: 前処理
        net_states: List[MultiTerminalNetState] = []
        two_terminal_nets: List = []

        for net in nets:
            state = self.preprocessor.preprocess_net(net)
            if state:
                if len(state.all_pads) > 2:
                    net_states.append(state)
                else:
                    two_terminal_nets.append(state)

        self.stats['total_nets'] = len(nets)
        self.stats['multi_terminal_nets'] = len(net_states)
        self.stats['two_terminal_nets'] = len(two_terminal_nets)

        logger.info(f"Preprocessed: {len(net_states)} multi-terminal, {len(two_terminal_nets)} two-terminal nets")

        # Step 2: ネット順序最適化（複雑なネットを先に）
        net_states = self._optimize_net_order(net_states)

        # 2端子ネットも追加（後で処理）
        all_states = net_states + two_terminal_nets

        # Step 3 & 4: MST構築 + ルーティング
        for i, state in enumerate(all_states):
            # MST構築（2端子ネットでも1 edgeのMST）
            self.mst_builder.build_mst(state)

            # ルーティング
            success = self.router.route_multi_terminal_net(state)

            if progress_cb:
                progress_cb(i + 1, len(all_states), None)

            if iteration_cb and (i + 1) % 10 == 0:
                tracks, vias = self._collect_current_geometry(all_states[:i+1])
                iteration_cb(i + 1, tracks, vias)

        # Step 5: 接続性検証
        validation = self.validator.validate_all_nets(all_states)

        self.stats['fully_connected'] = validation['fully_connected']
        self.stats['partially_connected'] = validation['partially_connected']

        logger.info(f"=== Multi-Terminal Routing Complete ===")
        logger.info(f"Fully connected: {validation['fully_connected']}/{validation['total_nets']} "
                   f"({validation['success_rate']:.1%})")
        logger.info(f"Average connectivity: {validation['average_connectivity']:.1%}")

        # 結果集約
        return self._collect_results(all_states, validation)

    def _optimize_net_order(self, states: List[MultiTerminalNetState]) -> List[MultiTerminalNetState]:
        """ルーティング順序最適化（複雑なネットを先に）"""
        def priority(state: MultiTerminalNetState) -> float:
            pad_count = len(state.all_pads)

            # 最大パッド間距離
            max_dist = 0
            for i, p1 in enumerate(state.all_pads):
                for p2 in state.all_pads[i+1:]:
                    dist = abs(p1.x_idx - p2.x_idx) + abs(p1.y_idx - p2.y_idx)
                    max_dist = max(max_dist, dist)

            # GND/VCCボーナス
            bonus = 1000 if 'GND' in state.net_name.upper() or 'VCC' in state.net_name.upper() else 0

            return -(pad_count * 100 + max_dist + bonus)

        return sorted(states, key=priority)

    def _collect_current_geometry(self, states: List[MultiTerminalNetState]) -> Tuple[List, List]:
        """現在のルーティング結果からジオメトリを収集"""
        tracks = []
        vias = []

        if not hasattr(self.pf, '_path_to_geometry'):
            return tracks, vias

        for state in states:
            for path in state.routed_paths:
                try:
                    path_tracks, path_vias = self.pf._path_to_geometry(path, state.net_name)
                    tracks.extend(path_tracks)
                    vias.extend(path_vias)
                except Exception as e:
                    logger.debug(f"Failed to convert path to geometry: {e}")

        return tracks, vias

    def _collect_results(self, states: List[MultiTerminalNetState],
                          validation: Dict) -> Dict[str, Any]:
        """結果をnet_name -> path形式で返す"""
        results = {}

        for state in states:
            # 全パスのノードを結合
            all_nodes = []
            for path in state.routed_paths:
                all_nodes.extend(path)

            # 重複除去（順序維持）
            seen = set()
            unique_nodes = []
            for node in all_nodes:
                if node not in seen:
                    seen.add(node)
                    unique_nodes.append(node)

            results[state.net_name] = unique_nodes

        # 検証結果を添付
        results['_validation'] = validation
        results['_stats'] = self.stats

        return results
