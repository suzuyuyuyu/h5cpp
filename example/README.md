# Examples

用途ごとに独立した最小例を置く。**ライブラリのビルドからは参照されない**。
テストのように必須にはせず、使い方をコンパクトに残すことが目的である。

| ディレクトリ | 内容 | 実行ファイル |
|---|---|---|
| `serial/` | 読み書き、文字列、属性、例外 | `example_serial` |
| `serial-interleaved/` | 成分ごとの配列をベクトル場として保存する | `example_serial_interleaved` |
| `parallel/` | 分散読み書きと担当範囲の取得 | `example_parallel` |
| `parallel-interleaved/` | 分散されたベクトル場（成分別配列 + 領域分割） | `example_parallel_interleaved` |

## ビルド

`example/` は**独立した CMake project** である。`find_package(h5cpp)` で
インストール済みのライブラリを使うので、**利用者が実際に書く CMake と
同じ形**になっている。

```sh
cmake -S example -B example/build/intel \
      -DCMAKE_PREFIX_PATH="$HOME/.local/opt/intel/h5cpp-0.1.0;$HOME/.local/opt/intel/h5c-0.1.0"
cmake --build example/build/intel --target examples
```

`h5cpp` がインストールされていなければ、隣のソースツリーを
`add_subdirectory` して一緒にビルドする。

`parallel/` は `H5C_ENABLE_PARALLEL=ON` でビルドした `h5c` を経由したときだけ
作られる。並列版の prefix を指す。

```sh
cmake -S example -B example/build/intel-mpi \
      -DCMAKE_PREFIX_PATH="$HOME/.local/opt/intel/h5cpp-mpi-0.1.0;$HOME/.local/opt/intel/h5c-mpi-0.1.0" \
      -DCMAKE_CXX_COMPILER=mpiicpx
cmake --build example/build/intel-mpi --target examples
```

## 実行

逐次の例はそのまま実行してよい。

```sh
cd example/build/intel && ./example_serial && ./example_serial_interleaved
```

**並列の例はログインノードで実行しないこと。** リポジトリのルートから投入する。

```sh
sbatch example/run-parallel-example.sh
```

## 出力例

```text
$ ./example_serial
/mesh/coords: rank=2 dims={3, 2} count=6
values: 1 2 3 4 5 6
time: 0.125
title: example field
units: m
root time attr: 0.125
exists /nope: false
missing dataset -> not found

$ ./example_serial_interleaved
components:
  1 10 100
  2 20 200
  3 30 300
  4 40 400
as stored: 1 10 100 2 20 200 3 30 300 4 40 400
v only: 10 20 30 40
ragged components -> shape mismatch

$ sbatch example/run-parallel-example.sh   # 4 ランク
rank 0: rows [0, 2)
rank 1: rows [2, 5)
rank 2: rows [5, 9)
rank 3: rows [9, 14)
global shape: {14, 3} from 4 ranks
partition: 0 2 5 9 14
```

## C++ 側の利点が見えるところ

`example_serial_interleaved` の `ragged components -> shape mismatch` は、長さの違う
成分を渡した場合である。C API は各成分の長さを知らないので `n` を検証できないが、
h5cpp は view を受け取るので**I/O を始める前に**弾ける。

`example_parallel` には、collective な呼び出しを `if (me == 0)` で囲むと
デッドロックするという注意をコメントで残してある。これは実際にこの例を書いた
ときに踏んだ誤りである。

## 分散されたベクトル場

`parallel-interleaved/` が実際のソルバーにいちばん近い。成分ごとの配列
（GPU に向いた持ち方）、MPI による領域分割、そして XDMF が要求する
`[total_n, 3]` の 1 データセットという 3 つを同時に満たす。

**ランク 0 はわざと 1 点も持たない。** 空の部分領域は正当であり、そのランクも
すべての collective 呼び出しに参加する。出力の `partition: 0 0 3 7 12` の
先頭が `0 0` になっているのがそれである。

```text
$ sbatch example/run-parallel-example.sh   # 4 ランク
rank 0: 0 points at rows [0, 0)
rank 1: 3 points at rows [0, 3)
rank 2: 4 points at rows [3, 7)
rank 3: 5 points at rows [7, 12)
global shape: {12, 3} from 4 ranks
partition: 0 0 3 7 12
as stored: 1000 2000 3000 1001 2001 3001 1002 2002 3002 2000 4000 6000 ...
```

`as stored` は、ランク 1 の 1 点目 `(u,v,w) = (1000, 2000, 3000)`、2 点目
`(1001, 2001, 3001)`、… と並んだあとにランク 2 の `(2000, 4000, 6000)` が続く。
成分が別々の配列であっても、ファイル上は点ごとにインターリーブされ、
各ランクのブロックが自分の offset に置かれていることが読み取れる。

なお属性は group ではなく `<path>/data` に付ける。group には
`__partition__` も入っているためである。
