import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
roberta = RobertaModel.from_pretrained(model_name_or_path='roberta.large.mnli',data_name_or_path='./data/CommonsenseQA')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('data/CommonsenseQA/valid.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            input = roberta.encode(
                'Q: ' + example['question']['stem'],
                'A: ' + choice['text'],
                no_separator=True
            )
            score = roberta.predict('mnli', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

print('Accuracy: ' + str(ncorrect / float(nsamples)))