from pretrain.gen_cooked_corpus import gen_cooked_corpus
from pretrain.gen_w2v import gen_w2c


if __name__ == "__main__":
    gen_cooked_corpus()
    print("熟语料生成OK！！！")
    gen_w2c()
    print("字向量生成OK！！！")
