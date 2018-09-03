## NLU项目
这个项目做得是实体的抽取和意图的分类，slot filling and intent classify

## 语料的处理
```
python gen_cooked_corpus_and_w2v.py
```
以上生成模型需要的语料，按1:2:13分别生成test数据、dev数据、train数据。以及用gensim生成词向量，这个可以在更大的语料中训练

## 训练
```
python train_evaluate.py --clean True --train True --model_type bilstm
```
上面用的是bilstm训练，也可以选择使用idcnn。

## 测试
```
python train_evaluate.py --train False
```