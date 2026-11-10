# 问卷转为 BERT 问答对

在项目根目录运行（仅使用 Python 标准库）：

```powershell
python dataset/scripts/create_bert_questionnaire_json.py
```

默认读取 `dataset/questionnaire/questionnaire_response_*.json`，输出
`dataset/preprocessed/questionnaire_bert.json`。支持通过 `--input-dir` 和
`--output` 指定其他路径。脚本从自身位置确定默认路径，因此也可从其他目录运行。

输出是 JSON 数组，每个元素对应一位受试者，包含字符串形式的患者编号
`subject_id`（保留前导零，例如 `001`）和 `qa_pairs`。`qa_pairs` 包含按题号
01–30 排列的 30 个问答对。受试者顺序与输入文件名的排序一致。每对仅包含：

- `question`：原始英文题目，未改写或翻译。
- `answer`：数值 1 / 0，分别对应原始答案 true / false。

脚本检查题数、题号完整性、重复受试者、题目一致性和答案类型。缺失或非布尔
答案会报错，不会自动填为 0。除患者编号和问答对外，不输出其他字段。

## 输入 BERT

JSON 存储可读文本；`input_ids`、`attention_mask` 和 `token_type_ids`
应由实际选用模型的 tokenizer 生成。下面以英文 BERT 为例，需要在训练环境中
安装 `transformers` 和 `torch`，并准备模型文件或允许首次下载。

```python
import json
from transformers import AutoTokenizer

with open("dataset/preprocessed/questionnaire_bert.json", encoding="utf-8") as f:
    samples = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
pairs = samples[0]["qa_pairs"]
inputs = tokenizer(
    [pair["question"] for pair in pairs],
    text_pair=[str(pair["answer"]) for pair in pairs],
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
# 每个张量形状为 [30, 128]；每道题形成：
# [CLS] question [SEP] 1/0 [SEP]
# 128 是此示例的长度配置，可按实际 tokenizer 和数据调整。
```

每位受试者对应 30 条序列。后续可将这 30 个问题的编码聚合为受试者表示。
划分训练集、验证集和测试集时应按 `subject_id` 以完整样本为单位，避免同一受试者的
不同题目进入不同集合。
