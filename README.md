[参考記事：Kaggle日記という戦い方(fkubotaさん)](https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068) 

[Github(fkubotaさん)](https://github.com/fkubota/kaggle-Cornell-Birdcall-Identification)

# 製造業対抗データインサイトチャレンジ2025
- result
  - public: 
  - private: 
  - rank: /



- directory tree
```
Kaggle-Cornell-Birdcall-Identification
├── README.md
├── data         <---- gitで管理するデータ
├── data_ignore  <---- .gitignoreに記述されているディレクトリ(モデルとか、特徴量とか、データセットとか)
├── nb           <---- jupyter lab で作業したノートブック
├── nb_download  <---- ダウンロードした公開されているkagglenb
└── src          <---- .ipynb 以外のコード

```

## Pipeline
- 実行例
  ```bash
  python3 pipeline.py --globals.balanced=1 --globals.comment=test
  ```

- 結果の表示例
  ```bash
  python3 show_result.py -d 0
  ```



## Info
- [issue board](https://github.com/fkubota/kaggle-Cornell-Birdcall-Identification/projects/1)   <---- これ大事だよ
- [google slide](https://docs.google.com/presentation/d/1ZcCSnXj2QoOmuIkcA-txJOuAlkLv4rSlS7_zDj90q6c/edit#slide=id.p)
- [flow chart](https://app.diagrams.net/#G1699QH9hrlRznMikAEAE2-3WTjreYcWck)
- [google drive](https://drive.google.com/drive/u/1/folders/1UDVIKTN1O7hTL9JYzt7ui3mNy_b6RHCV)
- ref:
  - [metricについて](https://www.kaggle.com/shonenkov/competition-metrics)
- docker run 時にいれるオプション
  - `--shm-size=5G`

## Dataset
|Name|Detail|Ref|
|---|---|---|
|SpectrogramDataset|5秒のSpectrogramを取得する。audiofileが5秒より短い場合、足りない部分に0 paddingする。5秒より長いものはランダムに5秒の部分を抽出する。|[公開ノートブック(tawaraさん)](https://www.kaggle.com/ttahara/training-birdsong-baseline-resnest50-fast)|
|SpectrogramEventRmsDataset|(バグ有り)SpectrogramDataset(SD)を改良。SDでは、鳥の鳴き声が入っていない部分を抽出する可能性があったのでそれを回避するために作った。librosa_rmsを使用し、バックグラウンドに比べてrmsが大きい値を取る時evet(birdcall)とした。|nb012|


## Event
|Name|Detail|Ref|
|---|---|---|
|nb017_event_rms|liborsaのrmsを使用。ラウドネスを見ていると思えばいい。|nb017|



## Paper
|No.|Status|Name|Detail|Date|Url|
|---|---|---|---|---|---|
|01|<font color='gray'>Done</font>|音響イベントと音響シーンの分析|日本語記事。まず最初に読むとよい。|2018|[url](https://www.jstage.jst.go.jp/article/jasj/74/4/74_198/_pdf)|
|02|<font color='green'>Doing</font>|PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition|アライさんがSEDの説明ノートブックで参照していた論文|201912|[url](https://arxiv.org/abs/1912.10211)|


## Memo


## Basics
**Overview**


### train.csv colomn infomaiton


|name|Explanation|
|----|----|
|rating|録音の質を表す(A,B,C,D,Eの5段階)|
|playback_sed|...|
|ebird_code|名前。nunique=264|
|channels|チャンネル数。2種類('1 (mono)', '2 (stereo)')|


## Log
### 20251019
- join!!
- 

### 2025


## コンペ反省会
|著者|Ref|
|---|---|
