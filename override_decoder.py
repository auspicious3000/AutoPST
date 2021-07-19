from onmt_modules.decoder_transformer import TransformerDecoder
from onmt_modules.misc import sequence_mask

  
class OnmtDecoder_1(TransformerDecoder):
    # overide forward
    # without teacher forcing for stop
    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)
        
        if step is None:
            tgt_lens = kwargs["tgt_lengths"]
        else:    
            tgt_words = kwargs["tgt_words"]
        
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        if step is None:
            tgt_max_len = tgt_lens.max()
            tgt_pad_mask = ~sequence_mask(tgt_lens, tgt_max_len).unsqueeze(1)
        else:    
            tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)

        with_align = kwargs.pop('with_align', False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns