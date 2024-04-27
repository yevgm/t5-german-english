# German to English translation using T5 pre-trained model

This project was done as part of an academic NLP course. We were given three [corpuses](/data) of training, validation and competition. The training and validation corpuses included German and English sentences, while the competition corpus had only German sentences. Our goal was to choose an appropriate architecture for this task, score a high sacreBLEU and submit the translation of the competition corpus to a competition between all students in the course. The Hebrew instructions for our project can be found [here](/hebrew-project-instructions.pdf).

We have chosen to use a Text-To-Text Transfer Transformer [(T5)](https://arxiv.org/abs/1910.10683) because it is a large language model that was pretrained on the C4 dataset of about 750GB of English text. Additionally, T5 was fine-tuned on diverse downstream tasks, including English to German translation. This fact ensured that the tokenizer of this model also included German tokens. The encoder-decoder architecture of T5 is particularly suited for translation tasks, where the encoder uses the full context of the source language sentence, while the decoder generates the target language's sentence in an auto-regressive way.


## Table of Contents

* [Method](#Method)
* [Setup](#setup)
* [Training](#training)
* [Inference](#Inference)
* [Results](#Results)


## Method
Our method includes the following steps:
<ul>
  <li>1. Dataset preparation:
    <ul>
    <li>1.1 The input training set is cleaned manually by removing incorrect translations. For example, German: -0,1, English: 0.1. </li>
    <li>1.2 Each dataset is parsed. Then, the training set is split into training (0.8) and validation (0.2), while the given validation set is used as a final test set.</li>
    <li>1.3 At the batching stage, we randomly sample batch_size examples and tokenize them using the SentencePiece tokenizer used by T5, with longest padding and a maximum context of 220.</li>
    </ul>
  </li>
  <li>2. Training step: The batches are passed to a pre-trained T5-base model, which is fine-tuned in a supervised way using the cross-entropy loss and AdamW optimizer.</li>
  <li>3. Evaluation step: 
  	<ul>
  		<li> 3.1 Prediction is auto-regressive and based on beam search. The maximum output length is 300 tokens. </li>
  		<li> 3.2 The final translations are evaluated with the sacreBLEU metric.</li>
  	</ul>
  </li>
</ul>

## Setup:
```bash
git clone https://github.com/yevgm/t5-german-english
cd t5-german-english
python3 -m venv venv
source ./venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install huggingface
pip install transformers
pip install evaluate
pip install sentencepiece
pip install accelerate -U
pip install sacrebleu
```

Our model's checkpoint can be found [here](https://drive.google.com/file/d/1_3b2sI3XX6CPTbdmM5LXCAZ071KMwoaD/view?usp=sharing).

## Training
T5-base model has 223M parameters. Thus, at least a single GPU is needed for training. We were able to train this model on a single GPU with 8 GB of RAM using the following settings:

| Hyperparameter  | Value        |
| -------------   | ----------- | 
| Batch size      | 4      | 
| Gradient accumulations steps      | 4      | 
| Learning rate   | 5e-5      |
| Epochs   | 40      |
| Mixed precision   | True      |
| Weight decay | 1e-4|
| Optimizer | AdamW |

We used 16-bit mixed precision and gradient accumulations steps to increasing the effective batch size to 16 and still fit in the memory of our GPU.

To run the training step, run the following command:
```bash
python ./main.py --save_model True --train True --debug False --epochs 40 --fp16 True --batch_size 4 --grad_accu_steps 4 --lr 5e-5
```

## Inference

| Hyperparameter  | Value        |
| -------------   | ----------- | 
| max_length      | 300      | 
| num_beams   | 8      | 
| early_stopping   | False      | 
| min_length   | 5      | 


To evaluate the model run the following command:
```bash
python ./main.py --train False --debug False --fp16 True --inference_batch_size 2
```

## Results

| Corpus        | SacreBLEU  |
| ------------- | ---------- | 
| Validation    | 38.09      | 

Example results from the validation corpus:

1.
	German example:
	> Und weiterreichende Kürzungen wie die von der EU vorgeschlagenen – 20 Prozent unterhalb der Werte von 1990 innerhalb von zwölf Jahren – würden die globalen Temperaturen bis 2100 lediglich um ein Sechzigstel Grad Celsius (ein Dreißigstel Grad Fahrenheit) senken, und das bei Kosten von 10 Billionen Dollar.
	Für jeden ausgegebenen Dollar hätten wir nur eine Wertschöpfung von vier Cent erreicht.

	Target translation:
	> And deeper emissions cuts like those proposed by the European Union – 20% below 1990 levels within 12 years – would reduce global temperatures by only one-sixtieth of one degree Celsius (one-thirtieth of one degree Fahrenheit) by 2100, at a cost of $10 trillion.
	For every dollar spent, we would do just four cents worth of good.

	Our translation:
	> And further cuts, such as those proposed by the EU – 20% below 1990 levels within 12 years – would lower global temperatures by just a sixth of a degree Celsius (a thirty-degree Fahrenheit) by 2100, at a cost of $10 trillion.For every dollar spent, we would have reached only a four-cent increase.


2.	
	German example:
	> Einige iranische Reformer und Exilanten haben die Wahl Ahmadinedschads mit dem Argument schön geredet, dass seine Regierung wahrscheinlich das wahre Gesicht der Regimes zeigt und sämtliche westliche Hoffnungen auf einen Kompromiss zerstört. Jedoch kann sich darin auch der Erfolg des Regimes widerspiegeln, die Unzufriedenheit über ein Vierteljahrhundert radikaler islamischer Herrschaft zu neutralisieren.
	Unabhängig vom Ausgang bedeutet Ahmadinedschads Sieg, dass alles, was mit dem Iran zu tun hat, voraussichtlich noch schwieriger wird.

	Target translation:
	> Some Iranian reformers and exiles put a bright face on Ahmadinejad’s election, arguing that his administration is more likely to show the regime’s real face and disabuse any Western hopes of compromise.
	Yet it may also represent the regime’s success at co-opting dissatisfaction with a quarter-century of radical Islamist rule.
	Whatever the outcome, for the West, Ahmadinejad’s victory means that everything related to Iran is set to become even hotter.

	Our translation:
	> Some Iranian reformists and exiles have summed up Ahmadinejad’s election with the argument that his government is likely to show the real face of the regime and destroy all Western hopes of a compromise.But it can also reflect the regime’s success in neutralizing discontent over a quarter-century of radical Islamic rule.Under Ahmadinejad’s victory, everything that Iran has to do is likely to become even more difficult.

