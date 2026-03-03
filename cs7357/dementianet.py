#sudo apt-get update; sudo apt-get install git-lfs; sudo git lfs install --system --skip-repo
#sudo mkdir -p /data/type3/data
#sudo chown -R $USER:$USER /data/type3/data
#place .env in working directory
#create dementianet conda environment
import os
import sys
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from packaging import version

import wandb

from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from random import randint

import transformers
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, EvalPrediction, Trainer, TrainingArguments
from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2PreTrainedModel, Wav2Vec2Model)
from transformers.file_utils import ModelOutput

import torchaudio
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from sklearn.model_selection import train_test_split
import IPython.display
import json

#if is_apex_available():
#    from apex import amp
#if version.parse(torch.__version__) >= version.parse("1.6"):
###
_is_native_amp_available = True
load_dotenv() # Loads the variables from .env
wandb_key  = os.getenv("WANDB_API_KEY")
token = os.getenv("HF_TOKEN")
wandb.login(key=wandb_key)
from huggingface_hub import login
login(token=token)

def split_df(df, col, val):
    return df[df[col] == val], df[df[col] != val]

def Audio(audio: np.ndarray, sr: int):
    """
    Use instead of IPython.display.Audio as a workaround for VS Code.
    `audio` is an array with shape (channels, samples) or just (samples,) for mono.
    """

    if np.ndim(audio) == 1:
        channels = [audio.tolist()]
    else:
        channels = audio.tolist()

    return IPython.display.HTML("""
        <script>
            if (!window.audioContext) {
                window.audioContext = new AudioContext();
                window.playAudio = function(audioChannels, sr) {
                    const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, sr);
                    for (let [channel, data] of audioChannels.entries()) {
                        buffer.copyToChannel(Float32Array.from(data), channel);
                    }

                    const source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start();
                }
            }
        </script>
        <button onclick="playAudio(%s, %s)">Play</button>
    """ % (json.dumps(channels), sr))

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]

def speech_to_array(path):
    speech, sr = torchaudio.load(path)
    transform = torchaudio.transforms.Resample(sr, 16000)
    speech = transform(speech)[0].numpy().squeeze()
    return random_subsample(speech, max_length=audio_len)


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label

def preprocess_fn(examples):
    speech_list = [speech_to_array(path) for path in examples[input_col]]
    target_list = [label_to_id(label, label_list) for label in examples[output_col]]
    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result['labels'] = list(target_list)
    result["input_length"] = [len(x) for x in result["input_values"]]

    return result

@dataclass
class SpeechClassifierModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """head for wav2vec classification task"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dense(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.post_init()
        #self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merge_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        elif mode == "sum":
            return torch.sum(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown merge strategy: {mode}")

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(input_values,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        hidden_states = outputs[0]
        hidden_states = self.merge_strategy(hidden_states, self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierModelOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"],} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.feature_extractor.pad( input_features,
                                    padding=self.padding,
                                    max_length=self.max_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                    return_tensors="pt",
        )

        batch['labels'] = torch.tensor(label_features, dtype=d_type)

        return batch

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    return {'accuracy': (preds == p.label_ids).astype(np.float32).mean().item()}

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

###

save_path = Path('/data/type3/data')

dm_path = save_path / 'dementia'
nd_path = save_path / 'nodementia'
dm_df = pd.read_csv(save_path/'dementia.csv')
nd_df = pd.read_csv(save_path/'nodementia.csv')
dm_df.head()

valid_dm, train_dm = split_df(dm_df, 'datasplit', 'valid')
test_dm, train_dm = split_df(train_dm, 'datasplit', 'test')
valid_nd, train_nd = split_df(nd_df, 'datasplit', 'valid')

train_dmlst = train_dm['name'].tolist()
train_ndlst = train_nd['name'].tolist()
valid_dmlst = valid_dm['name'].tolist()
valid_ndlst = valid_nd['name'].tolist()
print(len(train_dmlst), len(train_ndlst), len(valid_dmlst), len(valid_ndlst))

data_train = []
data_valid = []
for path in tqdm(dm_path.glob('**/*.wav')):
    name = str(path).split('/')[-1].split('.')[0]
    person = str(path).split('/')[-2]
    if person in train_dmlst:
        try:
            s = torchaudio.load(path)
            data_train.append({ 'file': name, 'label': 'dementia', 'path': path })
        except Exception as e:
            print(f'{path} is not a valid wav file', e)
            pass
    elif person in valid_dmlst:
        try:
            s = torchaudio.load(path)
            data_valid.append({ 'file': name, 'label': 'dementia', 'path': path })
        except Exception as e:
            print(f'{path} is not a valid wav file', e)
            pass

for path in tqdm(nd_path.glob('**/*.wav')):
    name = str(path).split('/')[-1].split('.')[0]
    person = str(path).split('/')[-2]

    if person in train_ndlst:
        try:
            s = torchaudio.load(path)
            data_train.append({ 'file': name, 'label': 'nodementia', 'path': path })
        except Exception as e:
            print(f'{path} is not a valid wav file', e)
            pass
    elif person in valid_ndlst:
        try:
            s = torchaudio.load(path)
            data_valid.append({ 'file': name, 'label': 'nodementia', 'path': path })
        except Exception as e:
            print(f'{path} is not a valid wav file', e)
            pass

train_df = pd.DataFrame(data_train)
valid_df = pd.DataFrame(data_valid)
train_df.head()

valid_df.head()
print("Labels: ", train_df.label.unique())
print(len(train_df.label.unique()))

train_df.groupby('label').count()[['path']]
print(f"train: {len(train_df)}")
print(f"valid: {len(valid_df)}")

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

train_df.to_csv(save_path / 'train_dm.csv', sep='\t', encoding='utf-8', index=False)
valid_df.to_csv(save_path / 'valid_dm.csv', sep='\t', encoding='utf-8', index=False)
###
data_files = {
    'train': '/data/type3/data/train_dm.csv',
    'valid': '/data/type3/data/valid_dm.csv'
}

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t"
)

train_data = dataset['train']
valid_data = dataset['valid']

print(train_data)
print(valid_data)

repo_name = "wav2vec2-large-xls-r-300m-dm32"

input_col = 'path'
output_col = 'label'
audio_len = 32

label_list = train_data.unique(output_col)
label_list.sort()
num_classes = len(label_list)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_list}")

model_name = "facebook/wav2vec2-xls-r-300m"
pooling_mode = "mean"

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_classes,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    )

setattr(config, "pooling_mode", pooling_mode)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name,)
target_sampling_rate = feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

train_data = train_data.map(preprocess_fn, batch_size=8, batched=True, num_proc=4,)
valid_data = valid_data.map(preprocess_fn, batch_size=8, batched=True, num_proc=4,)

data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name, config=config)
model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir = repo_name,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 2,
    eval_strategy = "steps",
    gradient_checkpointing = True,
    num_train_epochs = 22,
    logging_dir = None,
    save_steps = 136,
    eval_steps = 34,
    logging_steps = 136,
    learning_rate = 1e-4,
    save_total_limit = 2,
    fp16 = True,
    push_to_hub = True,
    #group_by_length = True,
    report_to = "none",
    warmup_steps = 0.1,
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=valid_data,
    processing_class=feature_extractor,
)

trainer.train()

trainer.push_to_hub(repo_name)