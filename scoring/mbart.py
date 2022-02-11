from torch import cuda
from utils import *
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast)

class mBART:
    def __init__(self, modelpath, tokenizerpath, device):
        self.model = MBartForConditionalGeneration.from_pretrained(modelpath)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizerpath)
        self.device = device
        self.model.to(self.device)

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang+"_XX"
        self.tokenizer.tgt_lang = tgt_lang+"_XX"
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device) #pt: PyTorch
        generated_tokens = self.model.generate(
            **encoded_input,
            num_beams=5,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang+"_XX"])
        return self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True)


if __name__ == "__main__":
    # define device
    device = "cuda:2" if cuda.is_available() else "cpu"
    
    # define model
    mbart = mBART(
        modelpath = "facebook/mbart-large-50-many-to-many-mmt",
        tokenizerpath = "facebook/mbart-large-50-many-to-many-mmt",
        device=device
    )
    
    # text = "Hello, my name is Adam."
    # hyp = mbart.translate(text=text, src_lang="en", tgt_lang="fr")
    # print(type(hyp), hyp)

    # group = "valid"
    # model_name = "mbart"
    # src_lang = "csw"
    # tgt_lang = "fr"
    # lang_pair = f"{src_lang}_{tgt_lang}"
    # src_file = f"/mnt/disks/adisk/data/fr_en/{lang_pair}/{group}.{src_lang}"
    # tgt_file = f"/mnt/disks/adisk/data/fr_en/{lang_pair}/{group}.{tgt_lang}"
    # predict(model=mbart,
    #         out_path=f"/mnt/disks/adisk/true_nmt/scoring/{model_name}/bidirectional",
    #         group=group,
    #         src_file=src_file,
    #         tgt_file=tgt_file,
    #         src_lang=f"{src_lang}_XX",
    #         tgt_lang=f"{tgt_lang}_XX",
    #         batch_size=16)

    group = "valid"
    model_name = "mbart"
    csw_file = f"/mnt/disks/adisk/data/fr_en/csw/{group}.src"
    tgt_file = f"/mnt/disks/adisk/data/fr_en/csw/{group}.tgt"
    dom_file = f"/mnt/disks/adisk/data/fr_en/csw/{group}.dom"
    predict_csw(model=mbart,
                out_path=f"/mnt/disks/adisk/true_nmt/scoring/{model_name}/csw",
                group=group,
                csw_file=csw_file,
                tgt_file=tgt_file,
                dom_file=dom_file,
                batch_size=16)

