# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from functools import cached_property

import torch


class VocabInfo:
    def __init__(self, vocab_map):
        self.name2val = vocab_map

        self.bos_id = vocab_map.get("<s>")
        self.eos_id = vocab_map.get("</s>")
        self.boi_id = vocab_map.get("<racm3:break>")
        self.eoi_id = vocab_map.get("<eoss>")
        self.pad_id = vocab_map.get("<pad>")
        self.eot_id = vocab_map.get("<reserved08706>")

    @property
    def begin_sequence(self):
        return self.bos_id

    @property
    def end_sequence(self):
        return self.eos_id

    @property
    def begin_image(self):
        return self.boi_id

    @property
    def end_image(self):
        return self.eoi_id

    @property
    def padding(self):
        return self.pad_id

    @property
    def end_turn(self):
        return self.eot_id

    @cached_property
    def val2name(self):
        return {v: k for k, v in self.name2val.items()}

    @cached_property
    def all_tokens(self):
        return sorted(self.name2val.values())

    @cached_property
    def image_tokens(self):
        return sorted([val for name, val in self.name2val.items() if name.startswith("IMGIMG")])

    @cached_property
    def special_tokens(self):
        return sorted([val for name, val in self.name2val.items() if name.startswith("<") and name != "<"])

    @cached_property
    def text_tokens(self):
        return sorted(set(self.all_tokens) - set(self.image_tokens) - set(self.special_tokens))


class VocabTranslation:
    def __init__(self, vocab_info: VocabInfo, device = None):
        self._vocab = vocab_info
        self._device = device

    @cached_property
    def bpe2img(self):
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name):
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

        return {tok: int(remap(self._vocab.val2name[tok])) for tok in self._vocab.image_tokens}

    @cached_property
    def img2bpe(self):
        return {v: k for k, v in self.bpe2img.items()}

    @cached_property
    def bpe2img_search_tensors(self):
        sorted_bpe = torch.tensor(sorted(self.bpe2img.keys()), device=self._device)
        sorted_img = torch.tensor(sorted(self.bpe2img.values()), device=self._device)
        return sorted_bpe, sorted_img

    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = torch.zeros(
            max(self.img2bpe.keys()) + 1,
            dtype=torch.int,
            device=self._device,
        )
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    def convert_bpe2img(self, bpe_batch):
        bpe_tok, img_tok = self.bpe2img_search_tensors
        return img_tok[torch.searchsorted(bpe_tok, bpe_batch)]

    def convert_img2bp2(self, img_batch):
        return self.img2bpe_mapping_tensor[img_batch]
