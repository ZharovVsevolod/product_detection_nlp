import torch
import einops
import numpy as np
import pandas as pd
import onnxruntime as ort
import youtokentome as yttm
from product_detection.data import text_lemmatize
from pymystem3 import Mystem
import re

from product_detection.config import KNOWN_CLASSES

def get_result(text: pd.Series) -> pd.Series:
    # Модель в формате onnx
    save_onnx = "weights/self_model.onnx"
    ort_session = ort.InferenceSession(save_onnx)
    answers = []

    # Предобработка
    np_text = text.to_numpy()

    # Лемматизатор
    mystem = Mystem()
    lemma_text = text_lemmatize(np_text, mystem)

    # Токенизатор
    tokenizer = yttm.BPE("weights/bpe_300.yttm")

    for phrase in lemma_text:
        phrase_in_tokens = text_preprocessing(phrase, tokenizer)
        phrase_in_tokens = einops.rearrange(phrase_in_tokens, "chunk -> 1 chunk")
        asnwer_class = get_answer_from_model(
            input_sample=phrase_in_tokens,
            ort_session=ort_session
        )
        answers.append(asnwer_class)

    return pd.Series(answers, name="class_predicted")

def text_preprocessing(text: np.array, tokenizer):
    # Очистка
    text = " ".join(text)
    text = re.findall("[а-яА-Я ]+", text)
    text = " ".join(text)

    # Токенизация
    text_tokens = tokenizer.encode(text, bos=True, eos=True)
    text_tokens = ensure_lenght(text_tokens)
    return np.array(text_tokens, dtype=np.int64)

def ensure_lenght(txt, chunk_lenght=60, pad_value=0):
        if len(txt) < chunk_lenght:
            txt = list(txt) + [pad_value]*(chunk_lenght - len(txt))
        else:
            txt = txt[:chunk_lenght]
        return txt

def get_answer_from_model(input_sample, ort_session: ort.InferenceSession) -> str:
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_sample}

    ort_outs = ort_session.run(None, ort_inputs)

    tensor_outputs = torch.from_numpy(np.array(ort_outs)).squeeze()[-1, :]
    answer_class = torch.argmax(tensor_outputs)
    return KNOWN_CLASSES[answer_class]