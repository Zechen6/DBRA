#!/bin/bash
for name in Ditto FedRep FedALA FedCAC SCAFFOLD; do
    echo "Starting $name at $(date)"
    nohup python attack/attack_conf_analysis.py "$name" > "cache/cifar10/attack_logs/${name}_AttackConf.log" 2>&1
    echo "Finished $name at $(date)"
done
echo "All tasks done at $(date)"