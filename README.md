# capture-delay-tools

QuEL-1 のキャプチャ遅延を調整するツールです。Qubex で波形を取得し、
sqe-analysis でパルスの位置を解析して、キャプチャ窓の中央に収まる遅延を求めます。

## セットアップ

```sh
uv sync --locked --python 3.13
```

## 使い方

- [日本語ノートブック](capture_delay.ipynb)
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
