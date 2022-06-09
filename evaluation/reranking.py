from transformers import GPT2Tokenizer, OPTForCausalLM, GPTJForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import os
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def gen(
        model, 
        tokenizer, 
        prompt, 
        input_device, 
        num_return_sequences = 5, 
        do_sample = True,
        max_length = 32,
        temperature = 0.9
    ):
    # inputs = tokenizer.encode(prompt, return_tensors="pt").to(input_device)
    # generation_output = model.generate(inputs, return_dict_in_generate = True)
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True,padding=True).to(input_device)
    generation_output = model.generate(
        inputs.input_ids,
        no_repeat_ngram_size = 3,
        temperature=temperature,
        max_length=max_length,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        output_scores = True,
        return_dict_in_generate = True
    )
    response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generation_output, response

@torch.no_grad()
def cal_log_perplexity_generate(generation_output):
    # print(generation_output.keys())
    sequence_num, generated_sequence_length = generation_output['sequences'].size()[0] ,len(generation_output['scores'])
    print("generate length:", generated_sequence_length)
    generation_output['scores'] = torch.stack(list(generation_output['scores']), dim=0)
    # print(generation_output['scores'].size())
    # print(f'sequence num = {sequence_num}, generated_sequence_length = {generated_sequence_length}')
    
    perp = []
    for i in range(sequence_num):
        generated_squence_ids = generation_output['sequences'][i][-generated_sequence_length:]
        scores = generation_output['scores'][:,i,:]
        log_softmax_scores = F.log_softmax(scores, dim=1)
        # print(scores.size(),log_softmax_scores.size())
        # print(scores[0][:10], log_softmax_scores[0][:10])
        assert scores.size()[0] == generated_squence_ids.size()[0]
        generated_squence_ids = generated_squence_ids.cpu().numpy()
        log_softmax_scores = log_softmax_scores.cpu().numpy()
        log_sum = 0
        for j in range(len(generated_squence_ids)):
            idx = generated_squence_ids[j]
            log_sum += log_softmax_scores[j][idx]
        perp.append(np.exp((-1/generated_sequence_length)*log_sum))
    return perp

def cal_log_perplexity_decode(logits, input_ids):
    perp = []
    for i in range(input_ids.size()[0]):
        generated_squence_ids = input_ids[i]
        n = generated_squence_ids.size()[0]
        scores = logits[i]
        log_softmax_scores = F.log_softmax(scores, dim=1)
        assert scores.size()[0] == n
        generated_squence_ids = generated_squence_ids.cpu().numpy()
        log_softmax_scores = log_softmax_scores.cpu().numpy()
        log_sum = 0
        for j in range(len(generated_squence_ids)):
            idx = generated_squence_ids[j]
            log_sum += log_softmax_scores[j][idx]
        perp.append(np.exp((-1/n)*log_sum))
    return perp


def rerank(input_text, tokenizer, model, input_device):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(input_device)
    outputs = model(**inputs)
    logits = outputs.logits
    perps = cal_log_perplexity_decode(logits, inputs.input_ids)
    assert len(perps) == len(input_text)
    ranked = sorted([ (input_text[i], perps[i]) for i in range(len(perps))], key = lambda x: x[1])
    return ranked


def main():
    ### set up device
    os.environ['CUDA_VISIBLE_DEVICES']="1,2,3,4,5,6,7,0"
    # main_model_name = "bigscience/T0-3B"
    main_model_name = "bigscience/T0"
    # main_model_name = "bigscience/T0p" # plus
    # main_model_name = "bigscience/T0pp" # plus plus

    ### set up tokenizer ###
    main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
    # main_tokenizer.pad_token = main_tokenizer.eos_token

    print(f"loading main model: {main_model_name}")
    ### set up model ###
    main_model = AutoModelForSeq2SeqLM.from_pretrained(main_model_name)
    main_model.parallelize()

    ### set input device as ###
    main_input_device = main_model.device
    print('model main device:',main_input_device)

    def load_prompts():
        retrieval_results = json.load(open())
    
    tempalte_names = [] 
    prompts = [] #TODO


    reranked_prompts = rerank(prompts, main_tokenizer, main_model, main_input_device)
    print(reranked_prompts)




if __name__ == "__main__":
    main()





    # ### set up device
    # os.environ['CUDA_VISIBLE_DEVICES']="1,2,3,4,5,6,7,0"
    # # main_model_name = "bigscience/T0-3B"
    # main_model_name = "bigscience/T0"
    # # main_model_name = "bigscience/T0p" # plus
    # # main_model_name = "bigscience/T0pp" # plus plus

    # ### set up tokenizer ###
    # main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
    # # main_tokenizer.pad_token = main_tokenizer.eos_token

    # print(f"loading main model: {main_model_name}")
    # ### set up model ###
    # main_model = AutoModelForSeq2SeqLM.from_pretrained(main_model_name)
    # main_model.parallelize()

    # ### set input device as ###
    # main_input_device = main_model.device
    # print('model main device:',main_input_device)

    # prompt = [
    #     "Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A:",
    #     "Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let\'s think step by step.",
    #     "Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let\'s think step by step. There are 16 balls in total. Half of the balls are golf balls. That means that there are 8 golf balls. Half of the golf balls are blue. That means the answer is",    
    #     "Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let\'s think step by step. There are 20 balls in total. Half of the balls are golf balls. That means that there are 10 golf balls. Half of the golf balls are blue. That means the answer is",   
    #     "Q: A juggler can juggle 20 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let\'s think step by step. There are 20 balls in total. Half of the balls are golf balls. That means that there are 10 golf balls. Half of the golf balls are blue. That means the answer is"    
    # ]

    # l = 256 # max length
    # k = 1 # num return sequence
    # t = 1.0 # temperature
    # do_sample = False

    # total_num = len(prompt)
    # prompt_perp = []
    # step_size = 32
    # steps = (total_num//step_size) + 1
    # for i in range(steps):
    #     start = i*step_size
    #     end = min((i+1)*step_size,total_num)
    #     input_text = prompt[start:end]
    #     inputs = main_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(main_input_device)
    #     outputs = main_model(**inputs)
    #     logits = outputs.logits
    #     prompt_perp += cal_log_perplexity_decode(logits, inputs.input_ids)
    # print('prompt:',prompt)
    # print("prompt log-perplexity:", prompt_perp)

    # ## generation
    # main_generation_output, main_responses = gen(
    #     main_model, 
    #     main_tokenizer, 
    #     prompt, 
    #     main_input_device, 
    #     num_return_sequences=k, 
    #     max_length=l, 
    #     temperature=t,
    #     do_sample=do_sample
    # )
    # main_perp = cal_log_perplexity_generate(main_generation_output)
    # print('generation log-perplexity:',main_perp)
    # print('prompt:', prompt,'\n')
    # print('response:')
    # for r in main_responses:
    #     print(r)
    #     print('\n')

    # torch.cuda.empty_cache()
    # del main_generation_output