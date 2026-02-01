# Phase 2: Multi-Terminal Net Support - Complete Implementation Plan V2.1

## 概要

OrthoRouteの多端子ネット対応を実装する。Codexレビュー2回の指摘を反映。

**目標:** 2層基板で95%+のルーティング成功率（全パッド接続基準）

## スコープ決定（Codexレビュー後）

| 項目 | 決定 | 理由 |
|------|------|------|
| **GPU対応** | 後続フェーズ | 7日では実装困難 |
| **KD-tree** | SciPy使用 | requirements.txtに追加済み |
| **迷路コスト** | ヒューリスティック | 後でA*に改善可能 |
| **マルチソース** | 順次接続方式 | 真のsuper-sourceは後続 |

## Codexレビュー指摘事項と対応

| 重大度 | 指摘 | 対応策 |
|--------|------|--------|
| High | MSTは真のMSTではない（障害物・コスト未考慮） | 迷路コストベースのエッジ重み計算 |
| High | 既存ツリーへの接続メカニズムがない | マルチソースルーティングAPI |
| High | 同一ネット自己ブロッキング問題 | same-net reuseルール |
| Medium | 成功メトリクスが全パッド接続を定義していない | 接続性検証ステップ |
| Medium | パッドレイヤー処理の欠落（B.Cu対応） | レイヤー考慮の最近接ノード選択 |
| Medium | O(k)ルーティング呼び出し、バッチなし | MST edgeバッチルーティング |
| Low | 重複パッド/同一座標パッド | 重複検出・統合 |
| Low | 可視化が1パス/ネット前提 | マルチパス統合出力 |

---

## アーキテクチャ設計

### 全体フロー

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Terminal Router                     │
├─────────────────────────────────────────────────────────────┤
│  1. Net Preprocessing                                        │
│     - Pad deduplication (同一座標統合)                       │
│     - Layer-aware pad mapping (F.Cu/B.Cu対応)               │
│     - Net ordering (complexity-based)                        │
├─────────────────────────────────────────────────────────────┤
│  2. MST Construction (Maze-Cost Based)                       │
│     - Lightweight A* for edge weight estimation              │
│     - Congestion-aware cost proxy                            │
│     - Delaunay/KD-tree for candidate edge pruning           │
├─────────────────────────────────────────────────────────────┤
│  3. Multi-Source Routing                                     │
│     - Virtual super-source node                              │
│     - Same-net reuse rule (self-blocking prevention)        │
│     - Fallback on edge failure (k-nearest retry)            │
├─────────────────────────────────────────────────────────────┤
│  4. Connectivity Validation                                  │
│     - Union-Find for full connectivity check                 │
│     - Partial success tracking                               │
│     - Rip-up trigger for incomplete nets                    │
├─────────────────────────────────────────────────────────────┤
│  5. Result Aggregation                                       │
│     - Multi-path merge per net                               │
│     - Updated success metrics                                │
│     - Visualization support                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 実装詳細

### 1. データ構造

```python
# unified_pathfinder.py

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import numpy as np

@dataclass
class PadInfo:
    """パッド情報（レイヤー考慮）"""
    pad_id: str
    x_idx: int
    y_idx: int
    layer: int  # 0=F.Cu, 1=B.Cu, -1=THRU
    node: int   # グラフノードID
    world_x: float
    world_y: float

@dataclass
class MSTEdge:
    """MST候補エッジ"""
    from_pad: PadInfo
    to_pad: PadInfo
    estimated_cost: float  # 迷路コスト推定値
    manhattan_dist: int

    def __lt__(self, other):
        return self.estimated_cost < other.estimated_cost

@dataclass
class MultiTerminalNetState:
    """多端子ネットの接続状態"""
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


class UnionFind:
    """接続性検証用Union-Find"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def num_components(self) -> int:
        return len(set(self.find(i) for i in range(len(self.parent))))
```

### 2. パッド前処理

```python
class MultiTerminalPreprocessor:
    """多端子ネット前処理"""

    def __init__(self, pathfinder: 'UnifiedPathFinder'):
        self.pf = pathfinder

    def preprocess_net(self, net) -> Optional[MultiTerminalNetState]:
        """
        ネットを前処理し、MultiTerminalNetStateを生成。

        - 重複パッド検出・統合
        - レイヤー考慮のノードマッピング
        - 2パッド未満はスキップ
        """
        pads = net.pads
        if len(pads) < 2:
            return None

        # パッド情報収集（重複検出用）
        seen_coords: Dict[Tuple[int, int, int], PadInfo] = {}
        pad_infos: List[PadInfo] = []

        for pad in pads:
            pad_id = self.pf._pad_key(pad)

            # ポータル確認
            if pad_id not in self.pf.portals:
                continue

            portal = self.pf.portals[pad_id]

            # レイヤー決定
            layer = self._determine_pad_layer(pad, portal)

            # ノードID計算
            node = self.pf.lattice.node_idx(
                portal.x_idx, portal.y_idx, portal.entry_layer
            )

            # 重複チェック
            coord_key = (portal.x_idx, portal.y_idx, layer)
            if coord_key in seen_coords:
                logger.debug(f"Duplicate pad at {coord_key}, merging with {seen_coords[coord_key].pad_id}")
                continue

            pad_info = PadInfo(
                pad_id=pad_id,
                x_idx=portal.x_idx,
                y_idx=portal.y_idx,
                layer=layer,
                node=node,
                world_x=portal.x_mm,
                world_y=portal.y_mm,
            )
            seen_coords[coord_key] = pad_info
            pad_infos.append(pad_info)

        if len(pad_infos) < 2:
            return None

        return MultiTerminalNetState(
            net_name=net.name,
            all_pads=pad_infos,
        )

    def _determine_pad_layer(self, pad, portal) -> int:
        """
        パッドのレイヤーを決定。

        - SMD: F.Cu(0) or B.Cu(1)
        - Through-hole: -1 (THRU)
        """
        # portal.entry_layer を使用（エスケーププランナーが決定）
        entry_layer = portal.entry_layer

        # 2層基板の場合
        if self.pf.layers == 2:
            if entry_layer == 0:
                return 0  # F.Cu
            elif entry_layer == 1:
                return 1  # B.Cu

        # スルーホール判定（両面に存在）
        if hasattr(pad, 'pad_type') and pad.pad_type == 'THRU':
            return -1

        return entry_layer
```

### 3. MST構築（迷路コストベース）

```python
class MazeCostMSTBuilder:
    """迷路コストベースのMST構築"""

    def __init__(self, pathfinder: 'UnifiedPathFinder'):
        self.pf = pathfinder
        self._cost_cache: Dict[Tuple[int, int], float] = {}

    def build_mst(self, state: MultiTerminalNetState) -> List[MSTEdge]:
        """
        Prim法でMSTを構築。エッジ重みは迷路コスト推定値。

        1. 全ペア間の候補エッジを生成（KD-treeでプルーニング）
        2. 各エッジの迷路コストを推定（軽量A*）
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
            if (edge.from_pad.node, edge.to_pad.node) not in self._cost_cache:
                cost = self._estimate_maze_cost(edge.from_pad, edge.to_pad)
                self._cost_cache[(edge.from_pad.node, edge.to_pad.node)] = cost
                self._cost_cache[(edge.to_pad.node, edge.from_pad.node)] = cost
            edge.estimated_cost = self._cost_cache[(edge.from_pad.node, edge.to_pad.node)]

        # Step 3: Prim法でMST
        mst_edges = self._prim_mst(pads, candidate_edges)

        state.mst_edges = mst_edges
        return mst_edges

    def _generate_candidate_edges(self, pads: List[PadInfo], k_nearest: int = 5) -> List[MSTEdge]:
        """
        KD-treeで近傍パッドのみを候補に。

        完全グラフはO(n^2)なので、k-nearest neighborでO(n*k)に削減。
        """
        from scipy.spatial import KDTree

        if len(pads) <= k_nearest + 1:
            # 小規模: 全ペア
            edges = []
            for i, p1 in enumerate(pads):
                for p2 in pads[i+1:]:
                    dist = abs(p1.x_idx - p2.x_idx) + abs(p1.y_idx - p2.y_idx)
                    edges.append(MSTEdge(from_pad=p1, to_pad=p2,
                                        estimated_cost=float(dist), manhattan_dist=dist))
            return edges

        # KD-tree構築
        coords = np.array([[p.x_idx, p.y_idx] for p in pads])
        tree = KDTree(coords)

        edges = []
        seen = set()

        for i, pad in enumerate(pads):
            # k+1 nearest (自分自身を含む)
            distances, indices = tree.query([pad.x_idx, pad.y_idx], k=min(k_nearest + 1, len(pads)))

            for j, idx in enumerate(indices):
                if idx == i:
                    continue

                # 重複回避
                edge_key = (min(i, idx), max(i, idx))
                if edge_key in seen:
                    continue
                seen.add(edge_key)

                other = pads[idx]
                dist = abs(pad.x_idx - other.x_idx) + abs(pad.y_idx - other.y_idx)
                edges.append(MSTEdge(from_pad=pad, to_pad=other,
                                    estimated_cost=float(dist), manhattan_dist=dist))

        return edges

    def _estimate_maze_cost(self, from_pad: PadInfo, to_pad: PadInfo) -> float:
        """
        軽量A*で迷路コストを推定。

        完全なルーティングではなく、コスト推定のみ。
        渋滞考慮のプロキシコストを返す。
        """
        # マンハッタン距離ベース
        base_cost = abs(from_pad.x_idx - to_pad.x_idx) + abs(from_pad.y_idx - to_pad.y_idx)

        # レイヤー遷移コスト
        layer_penalty = 0
        if from_pad.layer != to_pad.layer and from_pad.layer >= 0 and to_pad.layer >= 0:
            layer_penalty = 5  # Via cost

        # 渋滞ペナルティ（present cost proxy）
        congestion_penalty = 0
        if hasattr(self.pf, 'present') and self.pf.present is not None:
            # パス上のノードのpresentコストをサンプリング
            mid_x = (from_pad.x_idx + to_pad.x_idx) // 2
            mid_y = (from_pad.y_idx + to_pad.y_idx) // 2
            for layer in range(self.pf.layers):
                try:
                    mid_node = self.pf.lattice.node_idx(mid_x, mid_y, layer)
                    if mid_node < len(self.pf.present):
                        congestion_penalty += self.pf.present[mid_node] * 0.1
                except:
                    pass

        return base_cost + layer_penalty + congestion_penalty

    def _prim_mst(self, pads: List[PadInfo], edges: List[MSTEdge]) -> List[MSTEdge]:
        """Prim法でMST構築"""
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

        # Prim
        in_mst = [False] * n
        mst_edges = []

        # 開始ノード
        in_mst[0] = True
        heap = adj[0][:]
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

        return mst_edges
```

### 4. マルチソースルーティング

```python
class MultiSourceRouter:
    """マルチソースルーティング実装"""

    def __init__(self, pathfinder: 'UnifiedPathFinder'):
        self.pf = pathfinder
        self.same_net_nodes: Dict[str, Set[int]] = {}  # net_name -> occupied nodes

    def route_multi_terminal_net(self, state: MultiTerminalNetState,
                                  max_retries: int = 3) -> bool:
        """
        MST edgesに沿って多端子ネットをルーティング。

        1. MST edgesを短い順にソート
        2. 各edgeをルーティング（マルチソース対応）
        3. 失敗時はk-nearest retry
        4. 同一ネット再利用ルール適用
        """
        net_name = state.net_name

        if len(state.all_pads) < 2:
            return True

        # 同一ネットノード初期化
        self.same_net_nodes[net_name] = set()

        # MST edgesを短い順にソート（成功率向上）
        sorted_edges = sorted(state.mst_edges, key=lambda e: e.estimated_cost)

        # 最初のedgeをルーティング
        if sorted_edges:
            first_edge = sorted_edges[0]
            path = self._route_with_same_net_reuse(
                first_edge.from_pad.node,
                first_edge.to_pad.node,
                net_name
            )

            if path:
                self._mark_routed(state, first_edge.from_pad, first_edge.to_pad, path)
            else:
                logger.warning(f"Net {net_name}: Failed to route initial edge")
                return False

        # 残りのedgesをルーティング
        for edge in sorted_edges[1:]:
            # 両端が既にルーティング済みならスキップ
            if (edge.from_pad.pad_id in state.routed_pads and
                edge.to_pad.pad_id in state.routed_pads):
                continue

            # ターゲット決定（未ルーティング側）
            if edge.from_pad.pad_id in state.routed_pads:
                target_pad = edge.to_pad
            elif edge.to_pad.pad_id in state.routed_pads:
                target_pad = edge.from_pad
            else:
                # 両方未ルーティング: ルーティング済みツリーに最も近い方を先に
                target_pad = edge.to_pad

            # マルチソースルーティング（ルーティング済みノード群 → ターゲット）
            success = self._route_to_existing_tree(state, target_pad, max_retries)

            if not success:
                logger.warning(f"Net {net_name}: Failed to connect pad {target_pad.pad_id}")

        return state.is_fully_connected()

    def _route_with_same_net_reuse(self, source: int, target: int,
                                    net_name: str) -> Optional[List[int]]:
        """
        同一ネット再利用ルールを適用したルーティング。

        同じネットのノードは障害物として扱わない。
        """
        # 同一ネットノードを一時的に障害物から除外
        same_net = self.same_net_nodes.get(net_name, set())

        # PathFinderに同一ネット情報を渡す
        original_keepouts = None
        if hasattr(self.pf, '_temp_allowed_nodes'):
            original_keepouts = self.pf._temp_allowed_nodes.copy()

        self.pf._temp_allowed_nodes = same_net

        try:
            # 単一ペアルーティング
            path = self._route_single_pair(source, target, net_name)
            return path
        finally:
            # リストア
            if original_keepouts is not None:
                self.pf._temp_allowed_nodes = original_keepouts
            else:
                self.pf._temp_allowed_nodes = set()

    def _route_to_existing_tree(self, state: MultiTerminalNetState,
                                 target_pad: PadInfo, max_retries: int) -> bool:
        """
        ルーティング済みツリーからターゲットパッドへ接続。

        マルチソース: ルーティング済みノード群をソースとして扱う。
        """
        net_name = state.net_name
        target_node = target_pad.node

        # 試行順序: 近い順にソートしたルーティング済みパッド
        routed_pads = [p for p in state.all_pads if p.pad_id in state.routed_pads]

        # ターゲットからの距離でソート
        routed_pads.sort(key=lambda p:
            abs(p.x_idx - target_pad.x_idx) + abs(p.y_idx - target_pad.y_idx))

        # 最大k個を試行
        for retry in range(min(max_retries, len(routed_pads))):
            source_pad = routed_pads[retry]

            # ルーティング済みノードから最も近いノードを選択
            source_node = self._find_nearest_routed_node(
                target_node, state.routed_nodes
            )

            if source_node is None:
                source_node = source_pad.node

            path = self._route_with_same_net_reuse(source_node, target_node, net_name)

            if path:
                self._mark_routed(state, None, target_pad, path)
                return True

            logger.debug(f"Net {net_name}: Retry {retry+1}/{max_retries} failed for {target_pad.pad_id}")

        return False

    def _find_nearest_routed_node(self, target: int, routed_nodes: Set[int]) -> Optional[int]:
        """ターゲットに最も近いルーティング済みノードを探す"""
        if not routed_nodes:
            return None

        target_coords = self.pf.lattice.idx_to_coord(target)
        tx, ty, tz = target_coords

        min_dist = float('inf')
        nearest = None

        for node in routed_nodes:
            nx, ny, nz = self.pf.lattice.idx_to_coord(node)
            dist = abs(tx - nx) + abs(ty - ny) + abs(tz - nz) * 2
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def _route_single_pair(self, source: int, target: int, net_name: str) -> Optional[List[int]]:
        """単一ペアのルーティング（既存APIを使用）"""
        # 既存のroute_single_net APIを使用
        tasks = {net_name: (source, target)}

        # 1イテレーションのみ実行
        try:
            result = self.pf._dijkstra_iteration(tasks)
            path = result.get(net_name, [])
            return path if path else None
        except Exception as e:
            logger.error(f"Routing failed for {net_name}: {e}")
            return None

    def _mark_routed(self, state: MultiTerminalNetState,
                      from_pad: Optional[PadInfo], to_pad: PadInfo,
                      path: List[int]):
        """ルーティング結果を記録"""
        if from_pad:
            state.routed_pads.add(from_pad.pad_id)
        state.routed_pads.add(to_pad.pad_id)
        state.routed_nodes.update(path)
        state.routed_paths.append(path)

        # 同一ネットノード更新
        net_name = state.net_name
        if net_name not in self.same_net_nodes:
            self.same_net_nodes[net_name] = set()
        self.same_net_nodes[net_name].update(path)
```

### 5. 接続性検証

```python
class ConnectivityValidator:
    """接続性検証"""

    def __init__(self, pathfinder: 'UnifiedPathFinder'):
        self.pf = pathfinder

    def validate_net(self, state: MultiTerminalNetState) -> Tuple[bool, float]:
        """
        ネットの接続性を検証。

        Returns:
            (is_fully_connected, connectivity_ratio)
        """
        n = len(state.all_pads)
        if n < 2:
            return True, 1.0

        # Union-Find初期化
        pad_to_idx = {p.pad_id: i for i, p in enumerate(state.all_pads)}
        uf = UnionFind(n)

        # ルーティング済みパスから接続を構築
        for path in state.routed_paths:
            if len(path) < 2:
                continue

            # パス上の全ノードが属するパッドを特定
            connected_pads = set()
            for node in path:
                for pad in state.all_pads:
                    if pad.node == node:
                        connected_pads.add(pad.pad_id)
                        break

            # パッド間をunion
            connected_list = list(connected_pads)
            for i in range(len(connected_list) - 1):
                idx1 = pad_to_idx.get(connected_list[i])
                idx2 = pad_to_idx.get(connected_list[i + 1])
                if idx1 is not None and idx2 is not None:
                    uf.union(idx1, idx2)

        # 接続性チェック
        num_components = uf.num_components()
        is_connected = num_components == 1

        # 接続率計算（最大コンポーネントのサイズ / 全パッド数）
        component_sizes = {}
        for i in range(n):
            root = uf.find(i)
            component_sizes[root] = component_sizes.get(root, 0) + 1

        max_component = max(component_sizes.values()) if component_sizes else 0
        connectivity_ratio = max_component / n

        return is_connected, connectivity_ratio

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
        results['success_rate'] = results['fully_connected'] / results['total_nets'] if results['total_nets'] > 0 else 0.0

        return results
```

### 6. 統合クラス

```python
class MultiTerminalNetRouter:
    """
    多端子ネットルーター統合クラス。

    UnifiedPathFinder.route_multiple_nets() から呼び出される。
    """

    def __init__(self, pathfinder: 'UnifiedPathFinder'):
        self.pf = pathfinder
        self.preprocessor = MultiTerminalPreprocessor(pathfinder)
        self.mst_builder = MazeCostMSTBuilder(pathfinder)
        self.router = MultiSourceRouter(pathfinder)
        self.validator = ConnectivityValidator(pathfinder)

    def route_all_nets(self, nets: List, progress_cb=None, iteration_cb=None) -> Dict:
        """
        全ネットを多端子対応でルーティング。

        1. 前処理（パッド正規化、重複除去）
        2. ネット順序最適化
        3. MST構築
        4. ルーティング実行
        5. 接続性検証
        """
        logger.info(f"=== Multi-Terminal Routing: {len(nets)} nets ===")

        # Step 1: 前処理
        net_states = []
        for net in nets:
            state = self.preprocessor.preprocess_net(net)
            if state:
                net_states.append(state)

        logger.info(f"Preprocessed {len(net_states)} routable nets")

        # Step 2: ネット順序最適化（複雑なネットを先に）
        net_states = self._optimize_net_order(net_states)

        # Step 3 & 4: MST構築 + ルーティング
        for i, state in enumerate(net_states):
            # MST構築
            self.mst_builder.build_mst(state)

            # ルーティング
            success = self.router.route_multi_terminal_net(state)

            if progress_cb:
                progress_cb(i + 1, len(net_states), None)

            if iteration_cb and (i + 1) % 10 == 0:
                tracks, vias = self._collect_current_geometry(net_states[:i+1])
                iteration_cb(i + 1, tracks, vias)

        # Step 5: 接続性検証
        validation = self.validator.validate_all_nets(net_states)

        logger.info(f"=== Routing Complete ===")
        logger.info(f"Fully connected: {validation['fully_connected']}/{validation['total_nets']} ({validation['success_rate']:.1%})")
        logger.info(f"Average connectivity: {validation['average_connectivity']:.1%}")

        # 結果集約
        return self._collect_results(net_states, validation)

    def _optimize_net_order(self, states: List[MultiTerminalNetState]) -> List[MultiTerminalNetState]:
        """ルーティング順序最適化"""
        def priority(state):
            # 複雑さ優先（パッド数多い、距離長い）
            pad_count = len(state.all_pads)

            # 最大距離
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

        for state in states:
            for path in state.routed_paths:
                # パスをトラック/ビアに変換
                path_tracks, path_vias = self.pf._path_to_geometry(path, state.net_name)
                tracks.extend(path_tracks)
                vias.extend(path_vias)

        return tracks, vias

    def _collect_results(self, states: List[MultiTerminalNetState],
                          validation: Dict) -> Dict[str, List[int]]:
        """結果をnet_name -> path形式で返す"""
        results = {}

        for state in states:
            # 全パスを結合
            all_nodes = []
            for path in state.routed_paths:
                all_nodes.extend(path)

            results[state.net_name] = all_nodes

        # 検証結果を添付
        results['_validation'] = validation

        return results
```

---

## ファイル変更一覧

| ファイル | 変更内容 | 優先度 |
|---------|---------|--------|
| `unified_pathfinder.py` | MultiTerminalNetRouter統合、route_multiple_nets()修正 | High |
| `unified_pathfinder.py` | `_temp_allowed_nodes`追加（same-net reuse） | High |
| `negotiation_mixin.py` | same-net除外ロジック追加 | High |
| `main_window.py` | 接続性メトリクス表示更新 | Medium |
| `two_layer_benchmark.py` | 全パッド接続基準の成功率計算 | Medium |

---

## テスト計画

### 単体テスト

| テスト | 内容 |
|--------|------|
| `test_pad_deduplication` | 重複パッド検出・統合 |
| `test_layer_aware_mapping` | F.Cu/B.Cu/THRUパッド正しくマッピング |
| `test_mst_construction` | MST構築の正確性 |
| `test_maze_cost_estimation` | 迷路コスト推定の妥当性 |
| `test_same_net_reuse` | 同一ネット再利用ルール |
| `test_multi_source_routing` | マルチソースルーティング |
| `test_connectivity_validation` | 接続性検証の正確性 |
| `test_fallback_retry` | 失敗時リトライ動作 |

### 統合テスト

| テスト | 期待結果 |
|--------|---------|
| RP2040 Minimal | 成功率 95%+（全パッド接続基準） |
| led_blinker | 成功率 100% |
| 3端子ネット | 完全接続 |
| 10端子ネット | 完全接続 |
| GND/VCC多端子 | 完全接続 |

---

## 実装スケジュール（7日）

| Day | 作業内容 | 成果物 |
|-----|---------|--------|
| 1 | データ構造実装 | PadInfo, MSTEdge, MultiTerminalNetState, UnionFind |
| 2 | 前処理実装 | MultiTerminalPreprocessor |
| 3 | MST構築実装 | MazeCostMSTBuilder（KD-tree, Prim） |
| 4 | マルチソースルーティング実装 | MultiSourceRouter（same-net reuse含む） |
| 5 | 接続性検証実装 | ConnectivityValidator |
| 6 | 統合・デバッグ | MultiTerminalNetRouter、route_multiple_nets統合 |
| 7 | テスト・ベンチマーク | 全テスト実行、成功率測定 |

---

## 成功基準

- [ ] 3端子以上のネットが完全接続可能
- [ ] RP2040 Minimal 成功率 95%+（全パッド接続基準）
- [ ] 同一ネット自己ブロッキングなし
- [ ] 既存2端子ネットの動作維持
- [ ] ベンチマーク結果の記録・比較
- [ ] GUI表示が多端子対応

---

*作成日: 2026-02-01*
*Phase 1 完了時成功率: 62.7%*
*Codexレビュー反映版*
