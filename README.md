# training-a-chinese-model-for-spaCy-rasa
The step to train a chinese model for spaCy and RASA
--------------------------------------------------------
thanks to http://zhurongxin.site/2017/11/25/nlp_1_build_chinese_spacy_model_for_rasa/

spacy提供了便捷的工具，可以将生成的语言模型打包成独立的package，之后可以利用pip安装。
在rasa的配置文件中设定spacy_model_name字段为package的名字，rasa就可以实现自动导入。

实验环境

rasa_nlu: 0.10.4
spacy: 2.0.3
Anaconda: Anaconda3-5.0.1
python: 3.6.3
OS: win7
为spacy添加中文分词器(基于哈工大ltp工具包)
在spacy中，每种语言都对应一个包。因此我们需要进入中文对应的目录，添加分词器的代码。

修改spacy源码
以ANACONDA_PATH表示Anaconda安装路径

ANACONDA_PATH/Lib/site-package/spacy/lang/zh
修改__init__.py如下：

# coding: utf8
from __future__ import unicode_literals
from ...language import Language
from ...tokens import Doc
from ...attrs import LANG
import os
class ChineseTokenizer(object):
    def __init__(self, cls, nlp=None):
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
    def __call__(self, text):
        try:
            from pyltp import Segmentor
        except ImportError:
            raise ImportError("The Chinese tokenizer requires the pyltp library: "
                              "https://github.com/HIT-SCIR/pyltp")
        LTP_DATA_DIR = 'D:/path/to/ltp_data' # 这里需要改成哈工大语言模型ltp对应的路径
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        segmentor = Segmentor()  # 初始化实例
        segmentor.load(cws_model_path)  # 加载模型
        words = segmentor.segment(text)  # 分词
        words = [x for x in words if x]
        return Doc(self.vocab, words=words, spaces=[False] * len(words))
    # add dummy methods for to_bytes, from_bytes, to_disk and from_disk to
    # allow serialization (see #1557)
    def to_bytes(self, **exclude):
        return b''
    def from_bytes(self, bytes_data, **exclude):
        return self
    def to_disk(self, path, **exclude):
        return None
    def from_disk(self, path, **exclude):
        return self
class ChineseDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda text: 'zh'
    @classmethod
    def create_tokenizer(cls, nlp=None):
        return ChineseTokenizer(cls, nlp)
class Chinese(Language):
    lang = 'zh'
    Defaults = ChineseDefaults
    def make_doc(self, text):
        return self.tokenizer(text)
__all__ = ['Chinese']
安装pyltp工具
win + R呼出cmd，输入命令

pip install pyltp
下载pyltp分词模型
下载地址
解压后设置上面代码中LTP_DATA_DIR，指向解压出来的文件夹中的ltp_data目录

测试分词器
spacy采用pytest框架，所有的中文测试都在

ANACONDA_PATH/Lib/site-packages/spacy/tests/lang/zh
修改配置文件
spacy/tests/confests.py
添加中文代号’zh’

_languages = ['bn', 'da', 'de', 'en', 'es', 'fi', 'fr', 'ga', 'he', 'hu', 'id',
              'it', 'nb', 'nl', 'pl', 'pt', 'sv', 'xx', 'zh'] # zh需要自己添加
添加中文分词器

@pytest.fixture
def zh_tokenizer():
    return util.get_lang_class('zh').Defaults.create_tokenizer()
提供测试数据
spacy/tests/zh(如果没有，则创建)

添加__init__.py（空文件）
添加test_tokenizer.py如下
# coding: utf8
from __future__ import unicode_literals
import pytest
TOKENIZER_TESTS = [
        ("今天天气晴朗。", ['今天', '天气', '晴朗', '。']),
        ("我爱北京天安门。", ['我', '爱', '北京', '天安门', '。'])
]
@pytest.mark.parametrize('text,expected_tokens', TOKENIZER_TESTS)
def test_zh_tokenizer(zh_tokenizer, text, expected_tokens):
    tokens = [token.text for token in zh_tokenizer(text)]
    assert tokens == expected_tokens
运行测试
在spacy/tests/zh目录下，运行命令提示符，输入

py.test
pytest会自动运行test_tokenizer.py文件中的test_zh_tokenizer函数，并根据spacy/tests/confest.py中的fixture自动生成zh_tokenizer对象作为参数。

如果运行结果类似下
图：http://owruh8822.bkt.clouddn.com/spacy_1_tokenizer_test.png

分词器测试
那么恭喜你，分词器添加成功！

生成spacy中文模型
spacy官方提供的英语模型包括分词器，词典，命名实体识别模型，句法依赖分析模型，词性标注模型。由于只是简单示例，我们将要生成的中文语言模型只包含分词器和词典。

为中文模型构建词典
准备语料库
sample_corpus.txt，内容如下

你好  
你好啊  
你好吗  
hello  
hi  
早上好  
晚上好  
嗨  
是的  
是  
对的  
确实  
好  
ok  
好的  
好的，谢谢你  
对的  
好滴  
好啊  
我想找地方吃饭  
我想吃火锅啊  
找个吃拉面的店  
这附近哪里有吃麻辣烫的地方  
附近有什么好吃的地方吗  
肚子饿了，推荐一家吃饭的地儿呗  
带老婆孩子去哪里吃饭比较好  
想去一家有情调的餐厅  
bye  
再见  
886  
拜拜  
下次见  
感冒了怎么办  
我便秘了，该吃什么药  
我胃痛，该吃什么药？  
一直打喷嚏怎么办  
父母都有高血压，我会遗传吗  
我生病了  
头上烫烫的，感觉发烧了  
头很疼该怎么办  
减肥有什么好方法吗  
怎样良好的生活习惯才能预防生病呢
构建词典
复制以下代码，命名为gen_sample_vocab.py

# coding: utf8
from spacy.lang.zh import Chinese
from spacy.vocab import Vocab
from spacy.vectors import Vectors
import stringstore_test as st
import numpy as np
# ---------- Global Para ----------
# 创建语言类对象
nlp = Chinese()
#  构建分词器
tokenizer = Chinese().Defaults.create_tokenizer(nlp)
def gen_vocab(corpus_path):
    """构建示例vocab"""
    # ---------- StringStore ----------
    # 对样本语料切词，获取词列表
    word_set = set()
    with open(corpus_path) as file:
        for line in file.readlines():
            docs = tokenizer(line)
            words = [doc.text for doc in docs]
            word_set.update(words)
    word_list = [word for word in word_set]  # set => list
    
    # 利用词列表创建StringStore
    string_store = st.create_stringstore(word_list)
    # ---------- Vector ----------
    # 随机生成词向量，每个词用300维向量表示
    data = np.random.uniform(-1, 1, (len(word_list), 300))
    
    # vocab.vector.keys
    keys = word_list
    
    # 构建vectors
    vectors = Vectors(data=data, keys=keys)
    # ---------- Vocab ----------
    # 首先用StringStore创建词典，此时vocab.vectors无数据
    vocab = Vocab(strings=string_store)
    
    # 赋值vocab.vectors
    vocab.vectors = vectors
    return vocab
利用以下代码创建中文模型，并输出到文件
# coding: utf8
from spacy.language import Language
import gen_sample_vocab as gv
import os
"""
尝试新建中文language model，并输出到文件，供rasa使用。参考：https://spacy.io/api/language#to_disk
"""
# ------------------------ meta -------------------------------
"""语言模型配置字典"""
meta = {
  "lang": "zh",
  "pipeline": [
  ],
  "name": "sample",
  "license": "CC BY-SA 3.0",
  "author": "Rongxin Zhu",
  "url": "https://www.zhurongxin.site",
  "version": "0.0.0",
  "parent_package": "spacy",
  "vectors": {
    "keys": 90,
    "width": 300,
    "vectors": 300
  },
  "spacy_version": ">=2.0.0a18",
  "description": "Chinese sample model for spacy. Tokenizer and vocab only",
  "email": "731935354@qq.com"
}
# ------------------------ Vocab -----------------------------
vocab = gv.gen_vocab('data/corpus/sample_corpus.txt')
# ------------------------ Language -----------------------------
"""语言模型"""
nlp = Language(vocab=vocab, meta=meta)
# 输出语言模型
nlp.to_disk(path='data/model/zh')
运行上述代码之后，中文词典和分词器数据会保存在data/model/zh目录下
spacy模型结构

meta.json: 模型相关信息，包括模型的语言，作者，模型中包含哪些组件等等
tokenizer: 分词器相关数据
vocab: 词典相关的数据，结构如下
spacy词典结构
安装中文模型，供rasa使用
将包含语言模型的目录打包为可安装文件
在data/model目录下，打开cmd，输入命令：

python -m spacy package zh output
该命令会在根据meta.json，在output目录下生成zh_sample-0.0.0
如果报错找不到output路径，则需要新建output文件夹。

安装模型
cd output/zh_sample-0.0.0
python setup.py sdist
pip install dist/zh_sample-0.0.0.tar.gz
在rasa中使用中文模型
修改配置文件
在rasa的配置文件中，设置

"language": "zh"
"spacy_model_name" : "zh_sample"
由于我们生成的中文语言模型中不包含命名实体识别模型，因此pipeline中不能使用“ner_spacy”，可以用“ner_crf”代替

我的配置文件config_spacy_zh.json如下

{
  "name": "spacy_backend_zh_test",
  "pipeline": "spacy_sklearn",
  "language": "zh",
  "path" : "./projects",
  "data" : "data/demo-rasa_zh.json",
  "spacy_model_name" : "zh_sample"
}
spacy_sklearn是rasa预定义的pipeline模板之一，具体为

["nlp_spacy", "tokenizer_spacy", "intent_entity_featurizer_regex", "intent_featurizer_spacy", "ner_crf", "ner_synonyms",  "intent_classifier_sklearn"]
运行rasa_nlu
目录结构

learn_rasa
| -- sample_configs
     | -- config_spacy.json
训练rasa模型
在learn_rasa目录下运行cmd，输入命令

python –m rasa_nlu.train –c sample_configs/config_spacy_zh.json
基于示例训练数据训练rasa各个组件
示例数据目录在config_spacy_zh.json中配置，具体为data/demo-rasa_zh.json，形如

{
  "rasa_nlu_data": {
    "common_examples": [
      {
        "text": "你好",
        "intent": "greet",
        "entities": []
      },
      {
        "text": "你好啊",
        "intent": "greet",
        "entities": []
      }
    ]
  }
}
训练好的模型保存路径对应config_spacy_zh.json中的path字段。模型结构如下
rasa模型结构

crf_model.pkl: ner_crf模型
entity_synonyms.json: 同义词实体对照表
intent_classifier.pkl: 意图分类模型
metadata.json: 整体模型信息
training_data.json: 训练数据，与data/demo-rasa_zh.json相同
开启rasa后台
在learn_rasa目录运行cmd，输入命令

python –m rasa_nlu.server –c sample_configs/config_spacy_zh.json
利用浏览器测试
在浏览器中输入网址

http://localhost:5000/parse?q=胃痛怎么办
将返回如下结果
http://owruh8822.bkt.clouddn.com/spacy_1_result.png
示例结果
