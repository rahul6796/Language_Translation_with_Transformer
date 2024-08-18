
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from preprocessing import model_checkpoint
from preprocessing import tokenizer, tokenized_datasets

def load_model():
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt = True)
    return model


model = load_model()
data_collector = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors='tf')

batch = data_collector([tokenized_datasets["train"][i] for i in range(1, 3)])
# print(batch.keys())

tf_train_dataset = tokenized_datasets['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'label'],
    collate_fn=data_collector,
    shuffle=True,
    batch_size=128
)

tf_eval_dataset = tokenized_datasets['validation'].to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'label'],
    collate_fn=data_collector,
    shuffle=False,
    batch_size=128
)






