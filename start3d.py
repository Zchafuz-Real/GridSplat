import argparse

from src.runner3d import Runner
import time 
import torch
import optuna

test_config = {
    "name": "testing",
    "data": "../examples/data/hotdog/",
    "gaussians" : {
        "image_path" : "../../cactus2.jpg",
        "device": "cuda:0",
        "num_points": 30000,
        'bd': 2
    },
    "model" : {
        "model_type" : "dynamic",#specialized
        "L" : 3,
        "input_dim" : 3,
        "include_input" : True,
        "use_positional_encoding" : True,
        "input_encoding": "positional", #old positional fourier
        "min_freq" : 0,
        "max_freq" : 9.0,
        "n_freq" : 10,
        "scale": 10,
        "shared_hidden_dim" : 512,
        "hidden_dim" : 256,
        "num_layers": 6,
        "weight_init": True,
        "skip_connections": [1, 4],
    },
    "training": {
        "bd": 2,
        "viewer_enabled": True,
        "type": "test",
        "loss_function": "mse_ssim", #ssim_L1, L1, ssim, mse_loss, mse_ssim
        "lamba": 0.8, # there will be lamba ssim 
<<<<<<< HEAD
        "iterations": 300,
=======
        "iterations": 1000,
>>>>>>> 99fd3427c634af5be19b649682b6097f8cc109de
        "mean_lr": 0.01,
        "mlp_lr": 0.01,
        "device": "cuda:0",
        "allow_scheduler": False,
        "gamma": 0.5,
        "step_size": 9000,
        "allow_means_opptimization_iter": 0,
        "allow_sampling": True,
        "sample_every": 1,
        "strategy": "importance",
        "alpha": 0.05,
<<<<<<< HEAD
        "resolution": 64,
        "num_samples": 100000,
=======
        "resolution": 32,
        "num_samples": 40000,
>>>>>>> 99fd3427c634af5be19b649682b6097f8cc109de
        "filter": False,
        "randomize": True,
        "min_temp": 1.5,
        "max_temp": 3.0,
        "ema_decay": 0.9,
        "alt": False,
        "allow_logging": False,
        "factor": 3,
        "save_history": False,
        "background": "random",
        "image_type": "uint8",
        "save_model": True,
        "lrs": {
            "quat": 0.00089,
            #"rgb": 0.0000000001,
            "rgb": 0.000705,
            #"opacity": 0.001028,
            "opacity": 0.00000000000000000001,
            "scale": 0.0014,
            #"layers": 0.0006424
            "layers": 0.001
        },
    },
    "data_collector": {
        "type": "test",
        "loss_path": "losses",
        "img_path": "training",
        "means_path": "points",
        "num_of_gif_frames": 100,
        "num_of_xyz_frames": 100,
        "num_of_viewer_frames": 100,
        "log_all": False,
        "save_property": "resolution"
    },
    "data_loader": {
        "convert_NS_way": True,
        "background_color": "white", #this probably should always be white
        "image_type": "uint8"
    }
}


def normal_run(config_dir = None, data_dir = None):
    if config_dir is None:
        print("no config")
        config_dir = test_config
        config_dir["data"] = data_dir
    runner = Runner(config_dir)
    torch.cuda.empty_cache()
    start = time.time()
    loss = runner.run_experiments()
    print(f"Final loss: {loss}")
    end =  time.time() - start
    print(f"Time elapsed: {end} s, {end/60} min, {end/3600} h")
    print(f"Time per experiment: {end/len(runner.experiments)}")

def objective(trial):
    skip_placement = trial.suggest_int('skip_placement', 1, 4)
    test_config["model"]["skip_connections"] = [skip_placement]
    runner = Runner(test_config)
    torch.cuda.empty_cache()
    start = time.time()
    loss = runner.run_experiments(trial)
    end =  time.time() - start
    print(f"Time elapsed: {end} s, {end/60} min, {end/3600} h")
    print(f"Time per experiment: {end/len(runner.experiments)}")
    return loss

def evaluate_importance():
    study = optuna.load_study(study_name='layer_test', storage='sqlite:///layer_test.db')
    importance = optuna.importance.get_param_importances(study)
    print(importance)
    
def study_model(study_name):
    from optuna.trial import TrialState
    study = optuna.create_study(study_name=study_name, 
                                storage=f"sqlite:///{study_name}.db", 
                                load_if_exists=False, direction='minimize')
    
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    parser = argparse.ArgumentParser(description='Run the 3D grid splatting')
    parser.add_argument('--mode', choices = ["normal", "optuna", "importance"], type=str, default='normal', help='Mode of running the program')
    parser.add_argument('--study_name', type=str, help='Name of the Optuna study')    
    parser.add_argument('--config-dir', type=str, help='Directory of the config')
    parser.add_argument('--data-dir', type=str, help='Directory of the data')
    args = parser.parse_args()
    
    if args.mode == "normal":
        if args.config_dir is None and args.data_dir is None:
            parser.error("--data-dir is required for normal mode")
        elif args.config_dir is not None:
            normal_run(config_dir = args.config_dir)
        else:
            normal_run(config_dir = None, data_dir = args.data_dir)
    elif args.mode == "optuna":
        if args.study_name is None:
            parser.error("--study-name is required for optuna mode")
        study_model(args.study_name, args.storage)
    elif args.mode == "importance":
        if args.study_name is None:
            parser.error("--study-name is required for importance mode")
        evaluate_importance(args.study_name, args.storage)
        
if __name__ == "__main__":
    main()