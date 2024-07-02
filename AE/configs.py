config = {
    'env_name': "procgen", #"antmaze-xl-diverse-v0" , # "maze2d-medium-v1", #"antmaze-medium-diverse-v0", "maze2d-medium-v1",#"antmaze-large-diverse-v0", #"maze2d-medium-v1",# "ant-expert-v0",  # "walker2d-expert-v2", #
    'seed': 0,
    'reg': 0.1,
    'tanh': True,
    'traj_length': 5, #10,
    'batch_size': 50,
    'latent_dim': 8,
    'latent_reg': 0.1,
    'hidden_dims': [200, 200],
    'num_eval': 500,
    'eval_interval': 10,
    'train_epochs': 600,
    'goal_idxs': None,
    'render': True,
    'spirl': False
}

PATH_TO_AE_MODEL = "../models/AE_models/" + config['env_name'] + "-opal/600.pt"


# AE walker: 4051f2ce-9b05-48dd-80a8-617b026b869b
# AE maze2d medium: de24246b-e170-4d90-9d40-0d73b4d01029
# AE maze2d large: 5052e7d1-96eb-4897-aa54-a00df262c0cd
# AE antmaze medium: 377d5773-79a3-40a7-a202-5e28ddef5282
# AE antmaze large: 1bc6cad1-3f9e-449f-95cf-b38fd0014dac
