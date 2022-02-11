import os
from tqdm import tqdm
from collections import defaultdict


def predict(model, out_path, group, src_file, tgt_file, src_lang, tgt_lang, batch_size=16):
    """
    Uses the model to translate data found in the source file. And it writes
    the results in two files:
        - `out_path`/`group`.hyp: The translated sentences.
        - `out_path`/`group`.ref: The reference sentences.
    
    Note
    ----
    The input source/target files are both BPE encoded

    Parameters
    ----------
    model: transformers.Model
        The model used for translation.
    out_path: str
        The absolute/relative path where results will be found.
    group: str
        A string describing the group of data to be used.
        It's either "valid" or "test"
    src_file: str
        The absolute/relative path of the source file where each line
        represents a sentence.
    tgt_file: str
        The absolute/relative path of the target file where each line
        represents a sentence.
    src_lang: str
        The language code for the source sentence (e.g: en, fr, ...etc.).
    tgt_lang: str
        The language code for the target sentence (e.g: en, fr, ...etc.).
    batch_size: int
        The batch size (default=16). Using big batch size could lead to
        CUDA out-of-memory.
    """
    # create output directory if not exists
    os.makedirs(os.path.join(out_path, f"{src_lang}_{tgt_lang}"), exist_ok=True)
    
    # reading the files and decode it (remove `@@ `).
    with open(src_file, "r") as srcf, open(tgt_file, "r") as tgtf:
        src_lines = [sample.strip().replace("@@ ", "") for sample in srcf.readlines()]
        tgt_lines = [sample.strip().replace("@@ ", "") for sample in tgtf.readlines()]
    assert len(src_lines) == len(tgt_lines)
    
    # translate and write the results
    with open(os.path.join(out_path, group+f"-{tgt_lang}.hyp"), "a") as hypf, \
        open(os.path.join(out_path, group+f"-{tgt_lang}.ref"), "a") as reff:
        for batch_idx in tqdm(range(len(src_lines) // batch_size + 1), desc="Translating"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(src_lines))
            batch_src_lines = src_lines[start_idx:end_idx]
            batch_tgt_lines = tgt_lines[start_idx:end_idx]
            batch_hyp_lines = model.translate(batch_src_lines, src_lang, tgt_lang)
            for hyp, tgt in tqdm(zip(batch_hyp_lines, batch_tgt_lines)):
                hypf.write(hyp+"\n")
                reff.write(tgt+"\n")




def get_csw_src_tgt_lang(direction, dom):
    """
    Returns the source & target language using the direction tag
    and the dominant language with the help of this dictionary:
    `{"lang1": "fr", "lang2": "en"}`

    Parameters
    ----------
    direction: str
        The target language id (e.g: <2fr>, <2en>, ...).
    dom: str
        The dominant language as either "lang1" or "lang2".
    
    Returns
    -------
    (str, str):
        A tuple of two strings describing the source & target langauge.
    
    Example
    -------
    >>> src, tgt = get_csw_src_tgt_lang("<2fr>", "lang2")
    >>> src
    en
    >>> tgt
    fr
    """
    lang_d = {"lang1": "fr", "lang2": "en"}
    src = lang_d[dom.strip()]
    tgt = direction[2:-1]
    return src, tgt

def predict_csw(model, out_path, group, csw_file, tgt_file, dom_file, batch_size=16):
    """
    Uses the model to translate data found in the source file. And it writes
    the results in four files:
        - `out_path`/`group`-`lang1`.hyp: The translated lang1 sentences.
        - `out_path`/`group`-`lang2`.hyp: The translated lang2 sentences.
        - `out_path`/`group`-`lang1`.ref: The reference lang1 sentences.
        - `out_path`/`group`-`lang2`.ref: The reference lang2 sentences.
    
    Note
    ----
    The input files are both BPE encoded

    Parameters
    ----------
    model: transformers.Model
        The model used for translation.
    out_path: str
        The absolute/relative path where results will be found.
    group: str
        A string describing the group of data to be used.
        It's either "valid" or "test"
    csw_file: str
        The absolute/relative path of the code-switched file where each line
        represents a sentence.
    tgt_file: str
        The absolute/relative path of the target file where each line
        represents a sentence.
    dom_file: str
        The absolute/relative path of the dom file where each line represents
        the dominant language of that sentence in the source file.
    batch_size: int
        The batch size (default=16). Using big batch size could lead to
        CUDA out-of-memory.
    """
    # create output directory if not exists

    # reading the files and decode it (remove `@@ `).
    with open(csw_file) as cswf, open(tgt_file) as tgtf, open(dom_file) as domf:
        # group data based on dominant language
        src_data, tgt_data = defaultdict(list), defaultdict(list)
        for csw, tgt, dom in zip(cswf.readlines(), tgtf.readlines(), domf.readlines()):
            direction, _, src = csw.strip().partition(' ')
            src_lang, tgt_lang = get_csw_src_tgt_lang(direction, dom.strip())
            src_data[(src_lang, tgt_lang)].append(src.strip().replace('@@ ', ''))
            tgt_data[(src_lang, tgt_lang)].append(tgt.strip().replace('@@ ', ''))
    
    # make sure everything is ok
    for key in src_data.keys(): assert len(src_data[key]) == len(tgt_data[key])
    
    # translate and write the results
    for key in src_data.keys():
        src_lang, tgt_lang = key
        src_lines, tgt_lines = src_data[key], tgt_data[key]
        assert len(src_lines) == len(tgt_lines)
        hypf = open(os.path.join(out_path, group+f"-{tgt_lang}.hyp"), "a")
        reff = open(os.path.join(out_path, group+f"-{tgt_lang}.ref"), "a")
        for batch_idx in tqdm(range(len(src_lines) // batch_size + 1), desc=f"Translating from {src_lang} to {tgt_lang}"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(src_lines))
            batch_src_lines = src_lines[start_idx:end_idx]
            batch_tgt_lines = tgt_lines[start_idx:end_idx]
            batch_hyp_lines = model.translate(batch_src_lines, src_lang, tgt_lang)
            for hyp, tgt in tqdm(zip(batch_hyp_lines, batch_tgt_lines)):
                hypf.write(hyp+"\n")
                reff.write(tgt+"\n")
        hypf.close()
        reff.close()

