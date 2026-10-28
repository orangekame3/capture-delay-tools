# capture-delay-tools

QuEL-1・QuEL-3 のキャプチャ遅延を調整するツールです。Qubex で波形を取得し、
sqe-analysis でパルスの位置を解析して、キャプチャ窓の中央に収まる遅延を求めます。

## セットアップ

```sh
uv sync --locked --python 3.13
```

Qubex は [PR #387](https://github.com/amachino/qubex/pull/387) を最新upstreamへ統合した
`0bdd13c2e64d1af526bc6b927dd0b2c352d578f5` に固定しています。
QuEL-1 の合計遅延は 32 ns 刻みです。既存の `capture_delay_word` 設定も
非負整数かつ 4 の倍数である必要があります。QuEL-3 は `capture_delay_word` を使わず、
`capture_delay` をns値として指定・保存します。校正候補・最終値は両機種とも32 ns刻みです。

## 使い方

- [QuEL-1 日本語ノートブック](capture_delay.ipynb)
- [QuEL-3 日本語ノートブック](capture_delay_quel3.ipynb)
- [English notebook](capture_delay_en.ipynb)

ノートブックの実行環境に `.venv` を選び、リポジトリのルートから実行してください。
準備セルの設定ディレクトリ・対象 MUX・確認用周波数を実験環境に合わせて変更します。

最終波形を確認したら、候補を YAML に保存できます。

```python
calibrator.save_params("/path/to/params", overwrite=True)
```

API の使い方は [パッケージの説明](capture_delay_tools/README.md) を参照してください。

## 開発

```sh
uv run ruff check --fix
uv run ruff format
uv run pyright
uv run pytest
```
