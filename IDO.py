import subprocess
subprocess.run("pip install transformers")
subprocess.run("pip install datasets")
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def get(name,dataset1,dataset2,dataset3):
    model_name=name
    model=AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    d1=load_dataset(dataset1,split="train") 
    d2=load_dataset(dataset2,split="train")
    d3=load_dataset(dataset3,split="train")
    return model, tokenizer, d1, d2, d3

def preprocess(thinks, tokenizer):
    if 'text' in thinks:
        return tokenizer(thinks.get('text', ''),truncation=True,max_length=512)
    elif 'prompt' in thinks and 'completion' in thinks:
        return tokenizer(thinks.get('prompt', '')+" "+thinks.get('completion', ''),truncation=True,max_length=512)
    else:
        return{"input_ids":[],"attention_mask":[]}

def tokenize(d1, d2, d3, tokenizer):
    tokenized_dataset1=d1.map(lambda x: preprocess(x, tokenizer),batched=True)
    tokenized_dataset2=d2.map(lambda x: preprocess(x, tokenizer),batched=True)
    tokenized_dataset3=d3.map(lambda x: preprocess(x, tokenizer),batched=True)
    combined_dataset=concatenate_datasets([tokenized_dataset1,tokenized_dataset2,tokenized_dataset3])
    return combined_dataset

def train(model, tokenizer, combined_dataset, dirname,per_device_batch_size,num_train_epochs,save_steps,save_total_limit,learning_rate,modeldirname):
    training_args=TrainingArguments(
        output_dir=dirname,
        per_device_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        learning_rate=learning_rate
    )
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(modeldirname)

if __name__ == '__main__':
    modella=input("please give the model name")
    dta1=input("please give the first dataset")
    dta2=input("please give the second dataset")
    dta3=input("please give the third dataset")
    model, tokenizer, d1, d2, d3 = get(modella,dta1,dta2,dta3)
    
    combined_dataset = tokenize(d1, d2, d3, tokenizer)
    
    diras=input("please give the dirname of model")
    device_batch=int(input("please give the per_device_batch_size"))
    num=int(input("please give the num_train_epochs"))
    step=int(input("please give the save_steps"))
    total=int(input("please give the save_total_limit"))
    learning=float(input("please give the learning_rate"))
    mo=input("please give the modeldirname")
    train(model, tokenizer, combined_dataset, diras,device_batch,num,step,total,learning,mo)
