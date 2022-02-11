from torch import cuda
from utils import *
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer)

class M2M_100:
    def __init__(self, modelpath, tokenizerpath, device):
        self.model = M2M100ForConditionalGeneration.from_pretrained(modelpath)
        self.tokenizer = M2M100Tokenizer.from_pretrained(tokenizerpath)
        self.device = device
        self.model.to(self.device)

    def translate(self, text, src, tgt):
        self.tokenizer.src_lang = src
        self.tokenizer.tgt_lang = tgt
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device) #pt: PyTorch
        generated_tokens = self.model.generate(
            **encoded_input,
            forced_bos_token_id=self.tokenizer.get_lang_id(tgt))
        return self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True)


if __name__ == "__main__":
    # define device
    device = "cuda:3" if cuda.is_available() else "cpu"
    
    # define model
    m2m100 = M2M_100(
        modelpath = "facebook/m2m100_418M",
        tokenizerpath = "facebook/m2m100_418M",
        device=device
    )
    
    # text = "Hello, my name is Adam."
    # print(m2m100.translate(text = text, src = "en", tgt = "fr"))
    
    # group = "test"
    # model_name = "m2m100"
    # src_lang = "fr"
    # tgt_lang = "en"
    # lang_pair = f"{src_lang}_{tgt_lang}"
    # # lang_pair = "fr_en"
    # src_file = f"/mnt/disks/adisk/data/fr_en/{lang_pair}/{group}.{src_lang}"
    # tgt_file = f"/mnt/disks/adisk/data/fr_en/{lang_pair}/{group}.{tgt_lang}"
    # predict(model=m2m100,
    #         out_path=f"/mnt/disks/adisk/true_nmt/scoring/{model_name}/bidirectional",
    #         group=group,
    #         src_file=src_file,
    #         tgt_file=tgt_file,
    #         src_lang=f"{src_lang}",
    #         tgt_lang=f"{tgt_lang}",
    #         batch_size=16)

    group = "test"
    model_name = "m2m100"
    csw_file = f"/mnt/disks/adisk/data/fr_en/csw/{group}.src"
    tgt_file = f"/mnt/disks/adisk/data/fr_en/csw/{group}.tgt"
    dom_file = f"/mnt/disks/adisk/data/fr_en/csw/{group}.dom"
    predict_csw(model=m2m100,
                out_path=f"/mnt/disks/adisk/true_nmt/scoring/{model_name}/csw",
                group=group,
                csw_file=csw_file,
                tgt_file=tgt_file,
                dom_file=dom_file,
                batch_size=16)

