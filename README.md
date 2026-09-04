<h1 align="center">h5cpp</h1>

`h5c` の上に RAII、テンプレート、例外を載せた C++ ラッパーです。HDF5 を直接
呼ばず、すべて `h5c` の C API を経由します。これにより公式 HDF5 C++ インターフェースの
制約を受けずに並列 I/O が使えます。

## 必要なもの

- C++17 対応コンパイラ（C++20 も可、下記の注意を参照）
- [`h5c`](../h5c)（インストール済み、または `external/h5c` に配置）
- 並列 I/O を使う場合: MPI と、`H5C_ENABLE_PARALLEL=ON` でビルドした `h5c`

## Build

`h5c` はインストール済みのものを `find_package` で探します。見つからなければ
`external/h5c` または隣の `../h5c` を `add_subdirectory` します。

```sh
cp misc/CMakeUserPresets.json.template CMakeUserPresets.json
$EDITOR CMakeUserPresets.json      # h5c と HDF5 の場所を書く

cmake --preset my-intel
cmake --build --preset my-intel
ctest --preset my-intel
```

`CMakePresets.json` にはコンパイラと共通オプションだけを置き、**パスなど環境に
依存する値は書きません**。それらは `CMakeUserPresets.json`（Git 管理外）で
preset を継承して足します。テンプレートが `misc/` にあります。
ビルドディレクトリは `build/<preset 名>` に分かれます。

| preset | 用途 |
|---|---|
| `intel` / `intel-mpi` | Intel LLVM。優先して使うツールチェイン |
| `intel-cxx20` | C++20 言語モード（下記の注意を参照） |
| `gnu` / `gnu-mpi` | 可搬性の基準 |

インストール後は次の target を使います。HDF5 の include / library パスは
`h5c` の target が持っているので、利用側で重複記述する必要はありません。

```cmake
find_package(h5cpp CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE h5cpp::h5cpp)
```

## 使い方

```cpp
#include <h5cpp/h5cpp.hpp>

std::vector<double> values = {1, 2, 3, 4, 5, 6};

h5cpp::file f("result.h5", h5cpp::mode::truncate);
f.write("/rank/two", values, {3, 2});   // shape は {遅い次元, ..., 速い次元}
f.close();

h5cpp::file g("result.h5", h5cpp::mode::read);
auto got  = g.read<double>("/rank/two");      // 形状はファイルから決まる
auto meta = g.info("/rank/two");              // meta.dims == {3, 2}
```

失敗はすべて `h5cpp::error` として投げられ、`status()` で原因を判別できます。

```cpp
try {
    f.read<double>("/missing");
} catch (const h5cpp::error& e) {
    if (e.status() == H5C_ERR_NOT_FOUND) { /* ... */ }
}
```

`file` は move のみ可能で、デストラクタは例外を投げません。close の失敗を
観測したい場合は明示的に `close()` を呼んでください。

## Parallel (MPI)

並列 I/O は `h5cpp/h5cpp_mpi.hpp` にあります。**`<mpi.h>` を含むのはこのヘッダだけ**で、
`h5cpp/h5cpp.hpp` だけを使う限り MPI には一切依存しません。実装は `h5c_mpi.h` を
経由しており、HDF5 や MPI-IO を直接呼ぶことはありません。

ビルドには並列版 `h5c`（`H5C_ENABLE_PARALLEL=ON` でビルドしたもの）が必要です。

```sh
cmake --preset my-intel-mpi
cmake --build --preset my-intel-mpi
```

並列版 `h5c` の prefix は**共有ファイルシステム上**に置いてください。
`/tmp` はノードローカルなので、計算ノードから見えず `find_package` が失敗します。

`h5c` が逐次ビルドの場合は configure 時にエラーになります（target 上の
`H5C_HAVE_PARALLEL` を確認しています）。

並列テスト（`test_p*`）は `mpiexec` 経由で登録されています。**ログインノードでは
実行しないでください。** ジョブスクリプトからバッチ投入して実行します。

CTest のラベルは `mpi` で、`quick` には含まれていません。ログインノードで
習慣的に打つコマンドが誤って MPI ジョブを起動しないようにするためです。

```sh
ctest --preset my-intel-mpi   # 逐次テストのみ。ログインノードで可
```

```sh
# ジョブスクリプト内で（例: sbatch から）
ctest --test-dir build/my-intel-mpi -L mpi --output-on-failure
```

ランク数はジョブスクリプトの `--rsc p=N` で変えます
（テストは任意のランク数に追従します）。`-DH5CPP_TEST_NRANKS=<n>` は
ctest 登録側のランク数です。

```cpp
#include <h5cpp/h5cpp_mpi.hpp>

// shape は「ローカルブロック」。shape[0] が分割方向で、0 でも構いません。
std::vector<double> local(nlocal * 3);

h5cpp::parallel_file f("out.h5", h5cpp::mode::truncate);   // MPI_COMM_WORLD
f.write("/coords", local, {nlocal, 3});                    // 各ランクの担当分のみ

auto meta = f.info("/coords");     // meta.local.dims / meta.global.dims
auto got  = f.read<double>("/coords");   // 長さは __partition__ から決まる
f.close();
```

communicator と MPI-IO ヒントを明示する open、転送モードの切り替え、成分ごとに
分かれた配列のインタリーブ書き出しも使えます。

```cpp
h5cpp::parallel_file g("out.h5", h5cpp::mode::readwrite, comm, info);

g.set_collective(false);                 // 既定は collective。暗黙には切り替わりません
g.write_interleaved<double>("/velocity", {u, v, w});   // → [total, 3] の 1 データセット
g.read_interleaved<double>("/velocity", {u, v, w});
```

保存形式は h5fortran・`h5c` と同一です。パス `P` に書くと group ができ、
`P/data` に分割方向へ連結したデータ、`P/__partition__` に長さ `nprocs+1` の
ランク境界が入ります。

**すべての操作は collective です。** コンストラクタ・`close()`・`write` /
`read` / `info` / `set_collective` は全ランクが同じ順序で同じパスに対して
呼ぶ必要があります（`is_collective()`, `comm()`, `exists()` などの参照系は除く）。
引数の検証は `h5c` 側で全ランク集約されるため、1 ランクだけの不正な引数は
デッドロックせず全ランクで同じ `status()` の `h5cpp::error` になります。

## 可視化

`h5cpp/h5cpp_viz.hpp` は、分散メッシュと可視化フィールドを XDMF 用の配置で
書き出します。connectivity は各ランク内の 0-origin 節点番号を渡してください。

```cpp
#include <h5cpp/h5cpp_viz.hpp>

h5cpp::viz out("result/seq000000.h5", time);
h5cpp::viz_mesh mesh{h5cpp::viz_kind::unstructured, "fluid",
                      "Tetrahedron", 4, npoints, ncells};
out.begin_mesh(mesh);
out.write_nodes<double>({x, y, z});
out.write_connectivity(connectivity);
out.write_point_data<double>("Velocity", {u, v, w});
out.close();
```

すべての操作は collective で、ローカルの点数・セル数は 0 でも構いません。
ファイル配置の詳細は [`h5c/docs/FORMAT.md`](../h5c/docs/FORMAT.md) の
「可視化レイアウト」を参照してください。

## 配列の次元順序

`h5c` の規約をそのまま継承します。**shape は row-major**、すなわち
`shape.back()` が最も高速に変化します。

| 言語 | 宣言 | HDF5 dims |
|---|---|---|
| Fortran | `a(nx, ny)` | `[ny, nx]` |
| C++ | 平坦バッファ + `shape{ny, nx}` | `[ny, nx]` |

つまり **Fortran の `(nx, ny)` は C++ の `{ny, nx}`** です。転置もコピーも
発生しません。詳細は [`h5c/docs/FORMAT.md`](../h5c/docs/FORMAT.md) を参照してください。

## bool を扱うとき

`bool` ではなく **`h5c_bool_t`（= `int8_t`）** を使ってください。
`std::vector<bool>` はビットセットで連続バッファを持たないため、
HDF5 に渡せません。`type_of<bool>` は意図的に用意しておらず、
使うと理由を説明する `static_assert` が出ます。

```cpp
std::vector<h5c_bool_t> flags = {H5C_TRUE, H5C_FALSE, H5C_TRUE};
f.write("/flags", flags);
```

## C++20 を使う場合の注意

`H5CPP_CXX_STANDARD=20` を渡すと C++20 でコンパイルし、`std::span` が
利用可能なら `h5cpp::span` はそのエイリアスになります。利用できなければ
最小限の自前実装にフォールバックします。

**判定は `__cplusplus` ではなく `__cpp_lib_span` で行っています。**
これは必須です。たとえば Intel oneAPI の `icpx` はシステムの libstdc++ を
使うため、RHEL 8 の GCC 8.5 環境では `-std=c++20` を指定しても
`<span>` も `<version>` も存在しません。

```
c++17: uses fallback span
c++20: uses fallback span    ← icpx + GCC 8.5 の libstdc++
```

実際に `std::span` を使いたい場合は、新しい libstdc++ を持つ環境を
読み込んでください（例: `module load PrgEnvGCC/2023` で GCC 12.2）。
あるいは `icpx` に `--gcc-toolchain` を渡します。

## Examples

`example/` に最小の使用例を置いています。**ライブラリのビルドからは参照されない
独立した CMake project** で、`find_package` で利用するため、利用者が実際に書く
コードと同じ形になっています。

```sh
cmake -S example -B example/build/intel -DCMAKE_PREFIX_PATH=...
cmake --build example/build/intel --target examples
```

詳細は [example/README.md](example/README.md) を参照してください。

## ドキュメント

- 設計方針: [`../docs/h5c-h5cpp-design.md`](../docs/h5c-h5cpp-design.md)
- ファイルフォーマット仕様: [`../h5c/docs/FORMAT.md`](../h5c/docs/FORMAT.md)
