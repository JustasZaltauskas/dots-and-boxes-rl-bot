### Local Training Command
```
python ~/open_spiel/open_spiel/python/examples/alpha_zero.py --game "dots_and_boxes(num_rows=7,num_cols=7)" --path "repos/rl-project-KULeuven/az_checkpoints/dots_and_boxes_7x7/" --max_simulations 100 --train_batch_size 128 --replay_buffer_size 512 --eval_levels 3
```

### KULeuven Server Training Command
Activate the provided virtual environment
```
source /cw/lvs/NoCsBack/vakken/ac2223/H0T25A/ml-project/venv/bin/activate
```

Run the training script
```
python /cw/lvs/NoCsBack/vakken/ac2223/H0T25A/ml-project/software/open_spiel/open_spiel/python/examples/alpha_zero.py --game "dots_and_boxes(num_rows=7,num_cols=7)" --path "/home/r0959678/repos/rl-project-KULeuven/az_checkpoints/dots_and_boxes_7x7_actors4_sim75/" --max_simulations 75 --train_batch_size 128 --replay_buffer_size 512 --checkpoint_freq 50 --actors 4
```