# FedML


### lab3 standalong
```
# Download dataset
python3 create_femnist.py

python3 lab3.py \
--wandb_name fedavg_femnist_r20_test1 \
--gpu 0 \
--client_num_in_total 50 \
--client_num_per_round 50 \
--comm_round 200 \
--frequency_of_the_test 1 \
--epochs 1 \
--batch_size 16 \
--client_optimizer adam \
--lr 0.001 \
--ci 1 \
--seed 123
```

### lab3 rpc
```
WORKER_IDX=0

python3 lab3_rpc.py \
--wandb_name fedavg_femnist_r20_grpc_test2 \
--dataset femnist \
--data_dir ./femnist \
--client_num_in_total 4 \
--client_num_per_round 4 \
--batch_size 16 \
--client_optimizer adam \
--backend "GRPC" \
--lr 0.001 \
--epochs 1 \
--comm_round 200 \
--frequency_of_the_test 1 \
--gpu_mapping_file "./grpc_config/gpu_mapping.yaml" \
--gpu_mapping_key "mapping_FedML_gRPC_5" \
--grpc_ipconfig_path "./grpc_config/grpc_ipconfig.csv" \
--ci 1 \
--fl_worker_index "$WORKER_IDX"
```

$WORKER_IDX : first key 1~4 second 0