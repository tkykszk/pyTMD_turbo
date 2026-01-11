# pyTMD_turbo 実装状況

pyTMDの全機能をpyTMD_turboに実装するためのタスクリスト

## 凡例
- ✅ 実装済み
- ⚠️ 部分実装
- ❌ 未実装
- 🔄 pyTMD依存（独自実装が必要）

---

## 1. pyTMD.compute - 計算モジュール

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `tide_elevations` | 海洋潮汐高計算 | `tide_elevations()` | ✅ | - |
| `tide_currents` | 潮流計算 | `tide_currents()` | ✅ | - |
| `tide_masks` | 有効領域マスク | `tide_masks()` | ✅ | - |
| `LPET_elevations` | 長周期平衡潮汐 | `LPET_elevations()` | ✅ | - |
| `LPT_displacements` | 長周期潮汐変位 | - | ❌ | 中 |
| `OPT_displacements` | 海洋極潮汐変位 | - | ❌ | 中 |
| `SET_displacements` | 固体地球潮汐変位 | `SET_displacements()` | ✅ | - |
| `corrections` | 補正計算 | - | ❌ | 中 |

## 2. pyTMD.predict - 予測モジュール

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `time_series` | 時系列予測 | `predict_single/batch` | ✅ | - |
| `infer_minor` | マイナー成分推定 | `infer_minor()` | ✅ | - |
| `equilibrium_tide` | 平衡潮汐 | `equilibrium_tide()` | ✅ | - |
| `body_tide` | 天体潮汐 | `body_tide()` | ✅ | - |
| `solid_earth_tide` | 固体地球潮汐 | `solid_earth_tide()` | ✅ | - |
| `load_pole_tide` | 荷重極潮汐 | - | ❌ | 低 |
| `ocean_pole_tide` | 海洋極潮汐 | - | ❌ | 低 |
| `length_of_day` | 日長変化 | - | ❌ | 低 |

## 3. pyTMD.constituents - 潮汐成分

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `frequency` | 角周波数計算 | `frequency()` | ✅ | - |
| `arguments` | 天文引数 | `arguments()` | ✅ | - |
| `coefficients_table` | 係数テーブル | `coefficients_table()` | ✅ | - |
| `nodal` | 節点補正 | `nodal_modulation()` | ✅ | - |
| `nodal_modulation` | 節点変調 | `nodal_modulation()` | ✅ | - |
| `doodson_number` | Doodson番号 | - | ❌ | 低 |
| `minor_arguments` | マイナー成分引数 | `minor_arguments()` | ✅ | - |
| `group_modulation` | グループ変調 | - | ❌ | 低 |
| `aliasing_period` | エイリアシング周期 | - | ❌ | 低 |

## 4. pyTMD.io - 入出力モジュール

### 4.1 モデル管理

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `model` | モデルクラス | `model()` | ⚠️ | 高 |
| `load_database` | DB読み込み | `load_database()` | ✅ | - |
| `model.from_database` | DBからモデル生成 | `from_database()` | ✅ | - |
| `model.open_dataset` | データセット開く | `open_dataset()` | ⚠️ | 高 |
| `model.pathfinder` | パス解決 | - | ❌ | 中 |
| TMD accessor | xarray拡張 | `ds.tmd.interp/predict/infer` | ✅ | - |

### 4.2 フォーマット別リーダー

| フォーマット | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|-------------|-------|-------------|------|--------|
| OTIS binary | `io.OTIS` | `io.OTIS.open_dataset()` | ✅ | - |
| ATLAS compact | `io.ATLAS` | `io.ATLAS.open_dataset()` | ✅ | - |
| ATLAS netcdf | `io.ATLAS` | - | ❌ | 中 |
| GOT ascii | `io.GOT` | - | ❌ | 低 |
| GOT netcdf | `io.GOT` | `io.model` (部分) | ⚠️ | - |
| FES ascii | `io.FES` | - | ❌ | 低 |
| FES netcdf | `io.FES` | `io.FES.open_dataset()` | ✅ | - |
| TMD3 | `io.OTIS` | - | ❌ | 中 |

### 4.3 NOAA API

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `io.NOAA` | NOAA Webサービス | - | ❌ | 低 |

## 5. pyTMD.spatial - 空間計算

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `to_cartesian` | 地理→直交座標 | `to_cartesian()` | ✅ | - |
| `to_geodetic` | 直交→地理座標 | `to_geodetic()` | ✅ | - |
| `to_sphere` | 球面座標変換 | `to_sphere()` | ✅ | - |
| `convert_ellipsoid` | 楕円体変換 | `convert_ellipsoid()` | ✅ | - |
| `scale_factors` | スケール係数 | `scale_factors()` | ✅ | - |
| `datum` | 測地系定義 | `datum()` | ✅ | - |
| `to_ENU` | ENU座標変換 | - | ❌ | 低 |
| `from_ENU` | ENUからの変換 | - | ❌ | 低 |
| `compute_delta_h` | 高度差計算 | - | ❌ | 低 |

## 6. pyTMD.interpolate - 補間

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `interp1d` | 1次元補間 | SciPy使用 | ✅ | - |
| `extrapolate` | 外挿 | `extrapolate()` | ✅ | - |
| `inpaint` | 欠損値補間 | - | ❌ | 低 |

## 7. pyTMD.math - 数学関数

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `polynomial_sum` | 多項式和 | `polynomial_sum()` | ✅ | - |
| `normalize_angle` | 角度正規化 | `normalize_angle()` | ✅ | - |
| `rotate` | 回転行列 | `rotate_x/z()` | ⚠️ | - |
| `legendre` | ルジャンドル関数 | `legendre_polynomial()` | ✅ | - |
| `sph_harm` | 球面調和関数 | - | ❌ | 中 |
| `factorial` | 階乗 | - | ❌ | 低 |

## 8. pyTMD.astro - 天文計算

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `mean_longitudes` | 平均経度 | `mean_longitudes()` | ✅ | - |
| `lunar_ecef` | 月ECEF座標 | `lunar_ecef()` | ✅ | - |
| `solar_ecef` | 太陽ECEF座標 | `solar_ecef()` | ✅ | - |
| `gast` | グリニッジ恒星時 | `greenwich_mean_sidereal_time()` | ✅ | - |
| `lunar_approximate` | 月位置近似 | - | ❌ | 中 |
| `lunar_ephemerides` | 月暦 (JPL) | - | ❌ | 低 |
| `solar_ephemerides` | 太陽暦 (JPL) | - | ❌ | 低 |
| `doodson_arguments` | Doodson引数 | `doodson_arguments()` | ✅ | - |
| `delaunay_arguments` | Delaunay引数 | `delaunay_arguments()` | ✅ | - |

## 9. pyTMD.ellipse - 潮汐楕円

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `ellipse` | 潮汐楕円計算 | - | ❌ | 低 |
| `inverse` | 逆変換 | - | ❌ | 低 |

## 10. pyTMD.solve - 解析

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `constants` | 調和定数推定 | - | ❌ | 低 |

## 11. pyTMD.utilities - ユーティリティ

| 機能 | pyTMD | pyTMD_turbo | 状態 | 優先度 |
|------|-------|-------------|------|--------|
| `get_data_path` | データパス取得 | - | ❌ | 中 |
| `get_cache_path` | キャッシュパス | - | ❌ | 中 |
| `get_hash` | ハッシュ計算 | - | ❌ | 低 |
| ダウンロード機能 | HTTP/FTP | - | ❌ | 低 |

---

## 優先度別タスクサマリー

### 高優先度 (コア機能)

1. ~~**`tide_currents`** - 潮流計算の実装~~ ✅
2. ~~**`SET_displacements`** - 固体地球潮汐変位~~ ✅
3. ~~**`solid_earth_tide`** - 固体地球潮汐予測~~ ✅
4. ~~**`infer_minor`** - マイナー成分推定~~ ✅
5. ~~**TMD accessor** - xarray拡張 (`ds.tmd.interp()`, `ds.tmd.predict()`)~~ ✅
6. ~~**座標変換** - `to_cartesian`, `to_geodetic`~~ ✅
7. ~~**OTIS完全対応** - グリッド読み込み、トランスポート~~ ✅
8. ~~**ATLAS対応** - compact format~~ ✅

### 中優先度 (拡張機能)

9. ~~`tide_masks` - 有効領域マスク~~ ✅
10. ~~`LPET_elevations` - 長周期平衡潮汐~~ ✅
11. ~~`equilibrium_tide` - 平衡潮汐~~ ✅
12. ~~`extrapolate` - 外挿補間~~ ✅
13. ~~FES完全対応~~ ✅
14. ~~`normalize_angle`~~ ✅, `legendre`, `sph_harm`
15. ~~`doodson_arguments`, `delaunay_arguments`~~ ✅, `minor_arguments` ✅

### 低優先度 (特殊機能)

16. `load_pole_tide`, `ocean_pole_tide`
17. NOAA API
18. 潮汐楕円 (`ellipse`)
19. 調和定数推定 (`solve`)
20. ダウンロード機能

---

## 実装順序の提案

### Phase 1: コア機能完成 ✅
1. ✅ TMD accessor実装 (xarray拡張) - `ds.tmd.interp()`, `ds.tmd.predict()`, `ds.tmd.infer()`
2. ✅ 座標変換 (spatial) - `to_cartesian()`, `to_geodetic()`, `to_sphere()`, `scale_factors()`, `datum()`
3. ✅ infer_minor実装 - `infer_minor()`, `infer_diurnal()`, `infer_semi_diurnal()`
4. ✅ tide_currents実装 - `tide_currents()`

### Phase 2: フォーマット対応 ✅
5. ✅ OTIS完全対応 - `open_dataset()`, `open_otis_grid()`, `open_otis_elevation()`, `open_otis_transport()`, `open_mfdataset()`
6. ✅ ATLAS対応 - `open_atlas_grid()`, `open_atlas_elevation()`, `open_atlas_transport()`
7. ✅ FES完全対応 - `open_fes_elevation()`, `open_fes_transport()`, `open_dataset()`

### Phase 3: 固体地球潮汐 ✅
8. ✅ solid_earth_tide - `solid_earth_tide()`, Love numbers, ECEF変位計算
9. ✅ SET_displacements - `SET_displacements()` 高レベルラッパー
10. ✅ body_tide - `body_tide()` スペクトル法による潮汐計算

### Phase 4: 拡張機能 ✅
11. ✅ equilibrium_tide - `equilibrium_tide()`, Cartwright-Tayler-Edden法, Legendre多項式
12. ✅ tide_masks - `tide_masks()` モデル有効領域判定
13. ✅ extrapolate - `extrapolate()` k-d tree最近傍外挿, `bilinear()` 双線形補間

### Phase 5: 天文引数 ✅
14. ✅ normalize_angle - `normalize_angle()` 角度正規化
15. ✅ doodson_arguments - `doodson_arguments()` 6つのDoodson天文引数
16. ✅ delaunay_arguments - `delaunay_arguments()` 5つのDelaunay引数
17. ✅ schureman_arguments - `schureman_arguments()` FESモデル用Schureman引数
18. ✅ minor_arguments - `minor_arguments()` マイナー成分ノーダル補正

---

## 現在の実装率

| カテゴリ | 実装済み | 部分実装 | 未実装 | 実装率 |
|---------|---------|---------|--------|--------|
| compute | 5 | 0 | 3 | 62.5% |
| predict | 5 | 0 | 3 | 62.5% |
| constituents | 6 | 0 | 3 | 66.7% |
| io | 7 | 1 | ~10 | ~55% |
| spatial | 6 | 0 | 3 | 66.7% |
| interpolate | 2 | 0 | 1 | 66.7% |
| math | 3 | 1 | 2 | 50% |
| astro | 6 | 0 | 4 | 60% |
| **全体** | **40** | **2** | **~29** | **~58%** |

### 更新履歴
- Phase 5完了 (2026-01-11): 天文引数実装
  - `normalize_angle()`: 角度正規化（0〜circle範囲）
  - `doodson_arguments()`: 6つのDoodson天文引数（τ, S, H, P, N', Ps）
  - `delaunay_arguments()`: 5つのDelaunay引数（l, l', F, D, N）
  - `schureman_arguments()`: Schureman天文引数（I, ξ, ν, Qa, Qu, Ra, Ru, ν', ν''）
  - `minor_arguments()`: マイナー成分ノーダル補正（20成分）
  - パフォーマンス最適化: solar_ecef 7.2x, lunar_ecef 2.9x高速化
  - テスト: 23テスト追加 (test_phase5.py)

- Phase 4完了 (2026-01-11): 拡張機能実装
  - `equilibrium_tide()`: Cartwright-Tayler-Edden法による長周期平衡潮汐
  - `LPET_elevations()`: 高レベルラッパー
  - `tide_masks()`: モデル有効領域判定
  - `extrapolate()`: k-d tree最近傍外挿
  - `bilinear()`: 双線形補間
  - `mean_longitudes()`: 天文平均経度計算
  - `legendre_polynomial()`: 正規化ルジャンドル多項式
  - テスト: 27テスト追加 (test_phase4.py)

- Phase 3完了 (2026-01-11): 固体地球潮汐実装
  - `solid_earth_tide()`: IERS 2010準拠のECEF変位計算
  - `SET_displacements()`: 高レベルラッパー (地理座標入力、ENU/ECEF出力)
  - `body_tide()`: スペクトル法による潮汐計算
  - `love_numbers()`: 周波数依存Love数計算
  - `complex_love_numbers()`: マントル非弾性を含むLove数
  - テスト: 24テスト追加 (test_solid_earth.py)

- Phase 2完了 (2026-01-11): OTIS/ATLAS/FES フォーマット完全対応
  - OTIS: `open_dataset()`, `open_otis_grid()`, `open_otis_elevation()`, `open_otis_transport()`, `open_mfdataset()`
  - ATLAS: `open_atlas_grid()`, `open_atlas_elevation()`, `open_atlas_transport()`, `open_dataset()`
  - FES: `open_fes_elevation()`, `open_fes_transport()`, `open_dataset()`, `open_mfdataset()`
  - メモリマッピングサポート追加
  - xarray Dataset出力対応
