
from transformers import AutoTokenizer
from load_dataset import split_dataset

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensor = 'tf')


max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs  = [ex['en'] for ex in examples['translation']]
    targets = [ex['fr'] for ex in examples['translation']]

    model_input = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # setup the tokenizer for target 
    with tokenizer.as_target_tokenizer():
        lable = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_input['label'] = lable['input_ids']

    return model_input


split_datasets = split_dataset()

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched = True,
    remove_columns= split_datasets['train'].column_names

)
