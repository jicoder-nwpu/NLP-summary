#### vscode debug

##### .vscode/launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/train.py",    #执行的文件
            // "python": "/home/jhr/.virtualenvs/bart_tod/bin/python",  #同虚拟环境
            "console": "integratedTerminal",
            "args": [      #文件参数 有空格就切分
                "--mode=train",
                "--context_window=20",
                "--pretrained_checkpoint=./origin_model",
                "--gradient_accumulation_steps=8",
                "--cfg",
                "seed=557",
                "batch_size=2"
              ],
            //   "args": [
            //     "--mode=test",
            //     "--context_window=10",
            //     "--model_path=experiments/all_sd557_lr0.0006_bs2_sp5_dc0.8_cw20_model_./origin_model_1.0",
            //     "--gradient_accumulation_steps=8",
            //     "--cfg",
            //     "seed=557",
            //     "batch_size=4"
            //   ],
            "env": {"CUDA_VISIBLE_DEVICES":"0"},    #运行环境
            "justMyCode": false          #可以跳出本地代码 至库文件
        }
    ]
}
```

