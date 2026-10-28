# CaptureDelayCalibrator

QuEL-1・QuEL-3 のキャプチャ遅延を調整します。環境準備と実機ノートブックは [ルートの README](../README.md) を参照してください。

## 基本の流れ

```python
from capture_delay_tools import CaptureDelayCalibrator

calibrator = CaptureDelayCalibrator(
    exp, readout_frequencies_ghz={0: 11.0}, n_shots=1024
)
# QuEL-1: 128 ns coarse + 8 ns word (wordは4刻み)
calibrator.sweep({0: range(19)}, capture_delay_word={0: [0]})

# QuEL-3: nsを直接指定（校正は32 ns刻み）
# calibrator.sweep({0: np.arange(0, 2048, 32.0)})
calibrator.analyze()
calibrator.select()
calibrator.verify()
calibrator.finalize()
calibrator.check_final()
```

周波数は接続先の帯域に合わせて MUX ごとに指定します。11 GHz は記法の例です。
QuEL-1では MUX 内の1対象だけ送信し、全対象を同時取得します。
QuEL-3では `check_waveform` を1量子ビットずつ実行し、`readout_amplitude=1.0` を指定します。
掃引・再検証・最終確認で同じ取得方法を使い、測定後は設定を元に戻します。

## 遅延と候補選択

- QuEL-1では `capture_delay` は128 ns単位、`capture_delay_word` は8 ns単位で4の倍数です。
- QuEL-1の合計遅延は `128 * delay + 8 * word` nsです。
- QuEL-3では `capture_delay` にns値を直接指定します。0.8 nsグリッド外の値と `capture_delay_word` はエラーにします。
- 両機種とも校正候補・最終値は32 ns刻み、再検証の中心ずれ許容幅は±16 nsです。
- 両端の中点を窓中央に合わせます。片側だけの場合は、記録した readout duration から推定します。
- 各対象で異なる合計遅延の2点以上が整合することを要求します。
- 再検証では全対象の両端を実測し、前後32 ns以上の予測余白を持つ候補だけ採用します。

候補が得られない場合は `selection` / `final_report` の `reason` を確認してください。
`pulse_does_not_fit` は余白込みで収まらないことを示します。
掃引や解析をやり直すと、それ以降の候補・検証結果はクリアされます。

## 波形と図

```python
calibrator.plot_sweep()
calibrator.plot_analysis()
calibrator.plot_verification()
figures = calibrator.plot_final(show=False)  # 再測定せず、qubit ごとの Figure を取得

waveform = calibrator.waveforms[0].iq  # time_ns 座標付き xarray.DataArray
window = waveform.sel(time_ns=slice(100, 400))
```

解析範囲を限定する場合は `analyze(windows={"Q00": (0, 500)})` を使います。
解析結果・候補・検証結果は pandas DataFrame です。

## YAML 保存

```python
calibrator.save_params("/path/to/new/params")
calibrator.save_params("/path/to/existing/params", overwrite=True)
```

最終波形を確認してから保存してください。QuEL-1は `capture_delay.yaml`（ndelay）と
`capture_delay_word.yaml`（word）、QuEL-3は `capture_delay.yaml`（ns）のみを書き出します。既存 YAML をベースに、採用した MUX の値だけを更新します。
他の値・メタデータ・コメントは保持しますが、書式は変わる場合があります。
新しい保存先では Experiment の読み込み元を使い、`source_dir=` で変更できます。
保存しても実行中の Experiment の設定は再読み込みしません。
