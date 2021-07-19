from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    dim_neck_sea = 8,
    dim_freq_sea = 20,
    dim_spk = 82,
    dim_enc_sea = 512,
    chs_grp = 16,
    dim_freq_sp = 80,
  
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
