# GuideDogEstimation
1人称視点での盲導犬姿勢推定

盲導犬訓練の際、1人称視点での盲導犬の姿勢推定や、訓練士のハンドルの動きを計測する。

Pose Estimationの結果のJitterを抑える同時に、
リアルタイム性を確保するために、OneEuroFilterを適用してフィルタリングを行う。
