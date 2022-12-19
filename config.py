def get_config():
    config = {
        "seed" : 100, 
        "iteration" : 10000,
        "eval_step" : 256,
        "result_path" : "../result",
        "data" : {
            "dataset" : {
                "path" : "./fold.pkl"
            },
            "dataloader" : {
                "train" : {
                    "batch_size" : 16, 
                    "shuffle" : True, 
                    "num_workers" : 2,
                    "drop_last" : True, 
                    "pin_memory" : True
                }, 
                "val" : {
                    "batch_size" : 16, 
                    "shuffle" : False, 
                    "num_workers" : 2,
                    "drop_last" : True, 
                    "pin_memory" : True
                }, 
                "test" : {
                    "batch_size" : 1, 
                    "shuffle" : False, 
                    "num_workers" : 2,
                    "drop_last" : False, 
                    "pin_memory" : True
                }
            }
        },
        "model" : {
            "width" : 16,
            "blocks" : {
                "encoder" : [1, 1, 2, 4],
                "center" : 6,
                "decoder" : [1, 1, 1, 1]
            }
        },
        "optimizer" : {
            "Adam" : {
                "lr" : 1e-4
            }
        },
        "scheduler" : {
            "LinearLR" : {
                "start_factor": 1, 
                "total_iters": 1,
            },
        },
    }
    return config                                                                                                                                           