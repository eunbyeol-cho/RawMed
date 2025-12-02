from sacred import Experiment

ex = Experiment("METER", save_git_info=False)

@ex.config
def config():
    exp_name = "meter"
    # Data settings
    real_input_path = ""
    input_path = ""
    output_path = ""
    generated_data_path = ""

    ehr = ""
    embed_list = ["input", "type", "dpe"]
    test_subsets = "test"
    structure = "hi"
    seed = 0
    obs_size = 12
    max_event_size = 256
    max_event_token_len = 128

    input_index_size = 2320
    type_index_size = 7
    dpe_index_size = 15
    time_index_size = 14
    dpe_ignore_index = 0
    pad_token_id = 0

    time_type_token =  4
    time_dpe_token = 14
    time_data_type="text"
    time_len=2
    
    latent_dim = 1024
    spatial_dim = 4   

    #Model settings
    model = ""
    quantizer = ""
    encode_model = "cnn_encoder"
    decode_model = "cnn_decoder"
    require_gt_time = False

    embed_dim = 256
    n_heads = 4
    n_layers = 12
    dropout = 0.2

    #Training settings
    optimizer="AdamW"
    scheduler_type=None
    scheduler_step_size=30
    scheduler_gamma = 0.1
    lr = 5e-4
    n_epochs = 200
    batch_size = 16
    patience = 10

    
    # VQVAE settings
    drop_last_activation=False
    emb_codebooks = 256
    num_codebooks = 1024
    pad_token_id = num_codebooks
    commitment_cost = 1.0
    decay = 0.8
    num_quantizers = 1
    
    stochastic_sample_codes = True
    sample_codebook_temp = 0.1 
    shared_codebook= True
    lut_from_vqvae = False
    lut_learnable = True
    cumsum_depth_ctx = True

    pretrained_AE_path = None
    wandb_project_name = "25-05-23"
    test_only = False
    resume = False
    debug = True
    
    # Sampling
    temperature = 1
    time_temperature = 1
    topk=None
    sample = False
    decode_latent = False
    save_as_numpy = None
    load_from = ""
    
    # rq transformer
    input_embed_dim = 256
    vocab_size = 1024
    embd_pdrop = 0.2
    input_emb_vqvae = True
    head_emb_vqvae = True
    shared_tok_emb = True
    shared_cls_emb = True
    cumsum_depth_ctx = True
    block_size_cond = 1
    vocab_size_cond = 1
    mlp_bias = True
    attn_bias = True
    attn_pdrop = 0.0
    resid_pdrop = 0.1
    gelu = 'v1'
    time_cumsum=False
    time_index=0
    gen_samples=0


@ex.named_config
def task_train_VQVAE_indep():
    exp_name = "train_VQVAE_indep"
    model = "event_autoencoder"
    quantizer = "vector_quantizer"

    drop_last_activation = True
    batch_size = 4096

@ex.named_config
def task_test_VQVAE_indep():
    exp_name = "train_VQVAE_indep"
    model = "event_autoencoder"
    quantizer = "vector_quantizer"

    drop_last_activation = True
    batch_size = 16
    
    test_only = True
    debug = True
    embed_list = ["input", "type", "dpe"]
    require_gt_time=True
    save_as_numpy='input_logits,type_logits,dpe_logits,enc_indices'

@ex.named_config
def task_train_RQVAE_indep():
    exp_name = "train_RQVAE_indep"
    model = "event_autoencoder"
    quantizer = "residual_vector_quantizer"
    num_quantizers = 2
    
    drop_last_activation = True
    batch_size = 4096

@ex.named_config
def task_test_RQVAE_indep():
    exp_name = "train_RQVAE_indep"
    model = "event_autoencoder"
    quantizer = "residual_vector_quantizer"
    num_quantizers = 2
    
    drop_last_activation = True
    batch_size = 16
    
    test_only = True
    debug = True
    embed_list = ["input", "type", "dpe"]
    require_gt_time=True
    save_as_numpy='input_logits,type_logits,dpe_logits,enc_indices'


@ex.named_config
def task_train_AR():
    exp_name = "task_train_AR"
    embed_list = ["code", "time"]
    model = "event_transformer"
    batch_size = 64
    dropout = 0.1
    lr=3e-4


@ex.named_config
def task_test_AR():
    exp_name = "task_train_AR"
    embed_list = ["code", "time"]
    model = "event_transformer"
    batch_size = 64
    dropout = 0.1
    lr=3e-4

    test_only = True
    debug = True
    # require_gt_time = True
    require_gt_time = False
    save_as_numpy = 'input_logits,type_logits,dpe_logits,enc_indices,time_logits'


@ex.named_config
def task_sample_AR():
    exp_name = "task_train_AR"
    test_only = True
    embed_list = ["code", "time"]
    model = "event_transformer"
    batch_size = 64
    dropout = 0.1
    lr=3e-4

    sample = True
    test_only = True
    debug = True
    require_gt_time = False
    save_as_numpy = 'input_logits,type_logits,dpe_logits,enc_indices,time_logits'



