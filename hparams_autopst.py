from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    
    # sea params
    dim_neck_sea = 4,
    dim_freq_sea = 14,
    dim_enc_sea = 512,
    
    # autopst params
    dim_freq = 80,
    dim_code = 4,
    dim_spk = 82,
    dim_sty = 128,
    gate_threshold = 0.48,
    dec_steps_tx = 640,
    dec_steps_sp = 806,
    chs_grp = 16,
    
    # onmt params
    enc_layers = 4,
    enc_rnn_size = 256,
    dec_layers = 4, 
    dec_rnn_size = 256, 
    transformer_ff = 1024,
    heads = 8, 
    dropout = 0.1,
    attention_dropout = 0.1,  
    max_relative_positions = 0,
    copy_attn = False, 
    self_attn_type = "scaled-dot", 
    aan_useffn = False,
    full_context_alignment = False, 
    alignment_layer = 0,
    alignment_heads = 0,
    
    # pretrained model
    pretrained_path = './assets/xxx.ckpt',
    
    # data loader
    meta_file = './assets/train_vctk.meta',
    feat_dir_1 = './assets/vctk16-train-sp-mel',
    feat_dir_2 = './assets/vctk16-train-cep-mel',
    feat_dir_3 = './assets/vctk16-train-teacher', 
    batch_size = 4,
    shuffle = True,
    num_workers = 0,
    samplier = 2,
    max_len_pad = 2048,
    sampling_params = (0.8, 1.3, 0.1),
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)