# OrthoRoute Enhancement Plan for ChatPCB

## 概要

OrthoRouteをChatPCBの本命オートルーターとして強化する計画。
Freeroutingの将来的な廃止（KiCad 10でSWIG API削除予定）に備え、
KiCad IPC API nativeなOrthoRouteを2層基板で実用レベルに引き上げる。

## 現状分析

### 評価結果（2026-02-01）

| 項目 | 結果 |
|------|------|
| KiCad 9.0 + IPC API | 動作確認済み |
| 2層基板サポート | パッチ適用で動作 |
| RP2040 Minimal | 52ネット中36ネット成功（69%） |
| KiCadへの適用 | 116トラック + 42ビア 成功 |

### 発見された構造的問題

1. **多端子ネット未対応**（最重要）
   - `route_all_nets()` が最初の2パッドのみ接続
   - 3端子以上のネット（GND, VCC, バス等）は構造的に完全接続不可
   - **全層数で発生する問題**

2. **2層に不向きな設計**
   - F.Cu=H専用、B.Cu=V専用（Manhattan制約）
   - 外層平面配線禁止ガード
   - ポータルシステムが多層前提

3. **実行経路の分離**
   - GUI経路: ポータル未使用、layer=0固定
   - Headless経路: ポータル使用、多層対応

## 強化ロードマップ

### Phase 0: ベンチマーク基盤（1-2日）

**目的**: 改善効果を定量的に計測できる環境整備

- [ ] 2層テストボード選定（RP2040 Minimal, led_blinker）
- [ ] 成功率計測スクリプト作成
- [ ] ベースライン記録

**成果物**: `tests/benchmark/two_layer_benchmark.py`

### Phase 1: 2層専用モード（3-5日）

**目的**: 2層での自由度を最大化

#### 1.1 両軸許可（H/V自由）

**変更箇所**: `orthoroute/algorithms/manhattan/unified_pathfinder.py`

```python
# _assign_directions() の変更
def _assign_directions(self) -> List[str]:
    if self.layers == 2:
        return ['hv', 'hv']  # 両層でH/V自由
    # 多層は従来通り
    ...
```

**変更箇所**: `is_legal_planar_edge()` の拡張

```python
def is_legal_planar_edge(self, from_layer, direction):
    layer_dir = self.layer_dir[from_layer]
    if layer_dir == 'hv':
        return True  # 両方向許可
    return direction == layer_dir
```

#### 1.2 外層平面配線ガード解除

**変更箇所**: `unified_pathfinder.py` の `build_graph()`

```python
# 2層時は外層もエッジ生成対象
planar_layers = range(0, self.layers) if self.layers <= 2 else range(1, self.layers - 1)
```

#### 1.3 ポータル無効化（2層時）

```python
if board.layer_count <= 2:
    config.portal_enabled = False
```

**期待効果**: 成功率 69% → 85%+

### Phase 2: 多端子ネット対応（3-7日）

**目的**: 全パッドを接続可能にする（最重要改修）

**変更箇所**: `orthoroute/algorithms/manhattan/manhattan_router_rrg.py`

#### 2.1 MST型接続（最小実装）

```python
def route_all_nets(self, board, pathfinder):
    for net in board.nets:
        if len(net.pads) < 2:
            continue

        # 基準パッドを選定
        base_pad = net.pads[0]
        routed_pads = {base_pad}

        # 残りのパッドを順次接続
        for target_pad in net.pads[1:]:
            # 最も近いルーティング済みパッドを探す
            nearest = find_nearest_routed_pad(target_pad, routed_pads)

            # ルーティング実行
            success = pathfinder.route(nearest, target_pad)
            if success:
                routed_pads.add(target_pad)
```

#### 2.2 Steiner木最適化（中期）

- 既存配線への吸着
- 配線長最小化

**期待効果**: 成功率 85% → 95%+

### Phase 3: Rip-up強化（4-7日）

**目的**: 失敗ネットの回復率向上

- [ ] ネット順序最適化（長距離/多ピン優先）
- [ ] 局所Rip-up（衝突エリアのみ剥がし）
- [ ] present/historyバランス再調整（2層向け）

### Phase 4: 45度配線（1-3週、オプション）

**目的**: 混雑回避・配線長短縮

#### 4.1 短期: ポスト処理

- 直角L字を45度に置換
- DRCチェック付き

#### 4.2 中期: グラフ拡張

```python
# 対角エッジを許可
if layer_dir in ['hv', 'diagonal']:
    # dx=1, dy=1 のエッジを追加
    # コスト = sqrt(2) ≈ 1.414
```

## 実装方針

### ブランチ戦略

```
main (現状維持)
│
├── feature/two-layer-mode
│   ├── 両軸許可
│   ├── 外層ガード解除
│   └── ポータル無効化
│
├── feature/multi-terminal-net
│   └── MST型接続（2層・多層共通）
│
└── feature/rip-up-improvement
    └── ネット順序・局所Rip-up
```

### テスト戦略

| テストボード | 層数 | ネット数 | 用途 |
|-------------|------|---------|------|
| led_blinker | 2 | ~20 | 単体テスト |
| RP2040 Minimal | 2 | 52 | ベンチマーク |
| TestBackplane | 18 | 512 | 多層回帰テスト（GPU必要） |

### 成功基準

| Phase | 目標成功率 |
|-------|-----------|
| Phase 0 | ベースライン確立 |
| Phase 1 | 85%+ |
| Phase 2 | 95%+ |
| Phase 3 | 98%+ |

## リスクと対策

| リスク | 対策 |
|--------|------|
| KiCad IPC API不安定 | バージョン固定、エラーハンドリング強化 |
| 多層機能の破壊 | ブランチ分離、回帰テスト |
| 45度でDRC不整合 | Phase 4は後回し、Manhattan優先 |
| GPUなしで遅い | 2層は小規模なのでCPUで十分 |

## フォーク vs PR

**結論**: フォーク推奨

| 変更規模 | 推奨 |
|---------|------|
| 2層外層ガード解除 | PR可 |
| 両軸許可 | PR可 |
| 多端子ネット対応 | フォーク |
| 45度配線 | フォーク |

ChatPCB用にフォークし、小規模修正は上流にPRする方針。

## 次のアクション

1. [ ] フォークリポジトリ作成
2. [ ] Phase 0: ベンチマーク基盤構築
3. [ ] Phase 1: 2層専用モード実装
4. [ ] Phase 2: 多端子ネット対応

## 参考資料

- OrthoRoute Repository: https://github.com/bbenchoff/OrthoRoute
- KiCad IPC API: https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/
- kicad-python (kipy): https://github.com/KiCad/kicad-python

---

*作成日: 2026-02-01*
*作成者: ChatPCB Development Team*
