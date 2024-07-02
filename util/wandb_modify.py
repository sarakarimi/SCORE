import wandb

if __name__ == '__main__':
    api = wandb.Api()
    run = api.run("sara_team/ppo-skill-learning/fir3vh6j")
    run.config["exp_name"] = "ppo_test_test_test_more_test"
    run.update()