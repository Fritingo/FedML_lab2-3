# FedML


### lab2
```
# Download dataset
python3 create_shakespeare.py

python3 lab2.py \
--wandb_name fedavg_sha_l1_test2 \
--gpu 1 \
--client_num_in_total 10 \
--client_num_per_round 10 \
--comm_round 80 \
--frequency_of_the_test 1 \
--epochs 1 \
--batch_size 4 \
--client_optimizer adam \
--lr 0.001 \
--ci 1 \
--seed 123
```