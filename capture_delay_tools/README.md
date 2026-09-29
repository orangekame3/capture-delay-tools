# CaptureDelayCalibrator

QuEL-1 のキャプチャ遅延を調整します。環境準備と実機ノートブックは [ルートの README](../README.md) を参照してください。

## 基本の流れ

```python
from capture_delay_tools import CaptureDelayCalibrator

calibrator = CaptureDelayCalibrator(
    exp, readout_frequencies_ghz={0: 11.0}, n_shots=1024
)
calibrator.sweep({0: range(19)}, capture_delay_word={0: [0]})
calibrator.analyze()
calibrator.select()
calibrator.verify()
calibrator.finalize()
calibrator.check_final()
```

周波数は接続先の帯域に合わせて MUX ごとに指定します。11 GHz は記法の例です。
MUX 内では1対象だけ送信し、全対象を同時取得します。測定後は設定を元に戻します。

## 遅延と候補選択

- `capture_delay` は128 ns単位、`capture_delay_word` は8 ns単位で4の倍数です。
- 合計遅延は `128 * delay + 8 * word` ns。不正な入力値は丸めずエラーにします。
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

最終波形を確認してから保存してください。既存 YAML をベースに、採用した MUX の値だけを更新します。
他の値・メタデータ・コメントは保持しますが、書式は変わる場合があります。
新しい保存先では Experiment の読み込み元を使い、`source_dir=` で変更できます。
保存しても実行中の Experiment の設定は再読み込みしません。
