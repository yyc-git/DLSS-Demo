# How to install

项目根目录下运行：
```js
npm run bootstrap

cd packages/NSRR_inference/

npm run webpack:dev-server
```


## 运行说明

将会执行packages/NSRR_inference/src/main.ts代码，该代码使用WebNN实现推理，显示了 超采样后的高分辨率图片（目前只进行了快速训练和推理，所以效果不是很好）




关于训练部分，请详见packages/NSRR/

在推理时直接使用了packages/NSRR_inference/src/checkpoints/中保存的weight，它来自packages/NSRR/在训练后保存的weight(具体是保存在packages/NSRR/saved/checkpoints/中)