import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import re
import torch
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizerFast
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 정확한 오류 위치 추적[1][6]

def remove_prefix_and_normalize(text):
    import re
    text = re.sub(r"^(정답|답변|근거 문장|근거)\s*[:：]?\s*", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def replace_template_tags(text):
    tags = {
        "[인물]": "특정 인물",
        "[직업]": "특정 직업",
        "[장소]": "특정 장소",
        "[기업.기타]": "기타기업",
        "[제품]": "특정 제품",
        "[정당]": "특정 정당",
        "[종교]": "특정 종교",
        "[음식]": "특정 음식",
        "[기관]": "특정 기관",
        "[신념]": "특정 신념",
        "[이름]": "특정 이름",
        "[지역]": "특정 지역",
    }
    for k, v in tags.items():
        text = text.replace(k, v)
    return text

from konlpy.tag import Okt
okt = Okt()

def extract_key_tokens(text):
    return [w for w, p in okt.pos(text) if p in ['Noun', 'Verb', 'Adjective']]

stop_words = {'은', '는', '이', '가', '을', '를', '의'}

def remove_stopwords(tokens):
    return [w for w in tokens if w not in stop_words]

def normalize_symbols(text):
    text = re.sub(r"([ㅋㅎㅜㅠ])\1{2,}", r"\1\1", text)
    text = text.replace("?", " <QUESTION>").replace("!", " <EXCLAMATION>")
    return text

def full_preprocess(text):
    text = remove_prefix_and_normalize(text)
    text = replace_template_tags(text)
    text = normalize_symbols(text)
    tokens = extract_key_tokens(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)

# ✅ 1. 데이터 로드 & 전처리 적용
train_df = pd.read_csv(r"C:\Users\MSI\Desktop\SpamGuard-MLOps\data\train\merged_train.csv")
test_df = pd.read_csv(r"C:\Users\MSI\Desktop\SpamGuard-MLOps\data\test\merged_test.csv")

# 전처리 함수 적용 (full_preprocess는 네가 만든 함수 그대로)
train_df["text"] = train_df["text"].map(full_preprocess)
test_df["text"] = test_df["text"].map(full_preprocess)

# ✅ 2. Label Encoding
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["label"]).astype("int64")
test_df ["label"] = le.transform(test_df ["label"]).astype("int64")
num_labels = len(le.classes_)

# ✅ 3. Huggingface Dataset 변환
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

# ✅ 4. Tokenizer 로딩 및 토크나이징
MODEL_NAME = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)

if tokenizer.pad_token is None:                     # KoBERT엔 pad가 없음
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tok_fn(batch):
    return tokenizer(batch["text"],
               truncation=True,
               padding="max_length",
               max_length=128)

train_dataset = train_dataset.map(tok_fn, batched=True, remove_columns=["text"])
test_dataset  = test_dataset .map(tok_fn, batched=True, remove_columns=["text"])


# ✅ 5. 모델 로딩 (num_labels 중요!)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.resize_token_embeddings(len(tokenizer))       # pad 토큰 추가됐으므로 필수


data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
# ✅ 6. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"  # wandb 연동 X
)

# ✅ 7. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# ✅ 8. 학습 시작
trainer.train()

# ✅ 9. 예측 및 평가
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = test_df["label"].values

# ✅ 10. 모델 저장
model.save_pretrained("./saved_model/kobert")
tokenizer.save_pretrained("./saved_model/kobert")


print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
