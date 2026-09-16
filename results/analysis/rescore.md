# 전 체크포인트 소급 재채점 (STS-B dev Spearman)

## 조합별 상세 (run x entity x pooling)
| run | backbone | entity | pooling | none | center | center_pc1 | center_pc2 |
|---|---|---|---|---|---|---|---|
| r1a_t010_fixed | answerdotai/ModernBERT-base | student | first_last | 0.6177 | 0.6575 | 0.6792 | 0.6980 |
| r1a_t010_fixed | answerdotai/ModernBERT-base | student | last | 0.5891 | 0.6290 | 0.6617 | 0.6778 |
| r1a_t010_fixed_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6115 | 0.6513 | 0.6662 | 0.6966 |
| r1a_t010_fixed_seed43 | answerdotai/ModernBERT-base | student | last | 0.5765 | 0.6203 | 0.6443 | 0.6753 |
| r1b_t025_fixed | answerdotai/ModernBERT-base | student | first_last | 0.6250 | 0.6752 | 0.6907 | 0.7053 |
| r1b_t025_fixed | answerdotai/ModernBERT-base | student | last | 0.5994 | 0.6502 | 0.6766 | 0.6872 |
| r1b_t025_fixed_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6026 | 0.6509 | 0.6733 | 0.7036 |
| r1b_t025_fixed_seed43 | answerdotai/ModernBERT-base | student | last | 0.5727 | 0.6216 | 0.6557 | 0.6863 |
| r1c_t050_fixed | answerdotai/ModernBERT-base | student | first_last | 0.6290 | 0.6657 | 0.6799 | 0.7040 |
| r1c_t050_fixed | answerdotai/ModernBERT-base | student | last | 0.6085 | 0.6445 | 0.6675 | 0.6881 |
| r1c_t050_fixed_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6543 | 0.6784 | 0.6800 | 0.7045 |
| r1c_t050_fixed_seed43 | answerdotai/ModernBERT-base | student | last | 0.6410 | 0.6653 | 0.6709 | 0.6940 |
| r2_anchor_curriculum | answerdotai/ModernBERT-base | student | first_last | 0.5460 | 0.6239 | 0.6610 | 0.6849 |
| r2_anchor_curriculum | answerdotai/ModernBERT-base | student | last | 0.4840 | 0.5660 | 0.6098 | 0.6450 |
| r2_anchor_curriculum_ext3000 | answerdotai/ModernBERT-base | student | first_last | 0.5293 | 0.5831 | 0.6084 | 0.6084 |
| r2_anchor_curriculum_ext3000 | answerdotai/ModernBERT-base | student | last | 0.4968 | 0.5508 | 0.5791 | 0.5757 |
| r2_anchor_curriculum_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6069 | 0.6655 | 0.6719 | 0.7000 |
| r2_anchor_curriculum_seed43 | answerdotai/ModernBERT-base | student | last | 0.5781 | 0.6418 | 0.6546 | 0.6826 |
| r2_bert_anchor_curriculum | bert-base-uncased | student | first_last | 0.6348 | 0.6653 | 0.7070 | 0.7264 |
| r2_bert_anchor_curriculum | bert-base-uncased | student | last | 0.1070 | 0.1032 | 0.0927 | 0.0830 |
| r2_bert_anchor_curriculum_seed43 | bert-base-uncased | student | first_last | 0.6339 | 0.6704 | 0.7090 | 0.7307 |
| r2_bert_anchor_curriculum_seed43 | bert-base-uncased | student | last | 0.0231 | 0.0326 | 0.0309 | -0.0556 |
| r2_bert_lr1e4 | bert-base-uncased | student | first_last | 0.5969 | 0.6442 | 0.6693 | 0.6910 |
| r2_bert_lr1e4 | bert-base-uncased | student | last | 0.5265 | 0.5638 | 0.5923 | 0.5985 |
| r2_bert_lr3e5 | bert-base-uncased | student | first_last | 0.6192 | 0.6807 | 0.7033 | 0.7249 |
| r2_bert_lr3e5 | bert-base-uncased | student | last | 0.5718 | 0.6189 | 0.6489 | 0.6699 |
| r2_bert_lr3e5_warmup30 | bert-base-uncased | student | first_last | 0.6246 | 0.6848 | 0.7087 | 0.7267 |
| r2_bert_lr3e5_warmup30 | bert-base-uncased | student | last | 0.5801 | 0.6268 | 0.6573 | 0.6753 |
| r2_bert_lr5e5 | bert-base-uncased | student | first_last | 0.6146 | 0.6709 | 0.6913 | 0.7153 |
| r2_bert_lr5e5 | bert-base-uncased | student | last | 0.5626 | 0.6042 | 0.6333 | 0.6512 |
| r2_bert_uniform_push | bert-base-uncased | student | first_last | 0.6235 | 0.6848 | 0.7099 | 0.7273 |
| r2_bert_uniform_push | bert-base-uncased | student | last | 0.5784 | 0.6277 | 0.6585 | 0.6773 |
| r2_bert_uniform_push | bert-base-uncased | teacher | first_last | 0.6326 | 0.6913 | 0.7185 | 0.7338 |
| r2_bert_uniform_push | bert-base-uncased | teacher | last | 0.5912 | 0.6391 | 0.6713 | 0.6894 |
| r2_bert_uniform_push_optuna_t0 | bert-base-uncased | student | first_last | 0.6404 | 0.6972 | 0.7283 | 0.7414 |
| r2_bert_uniform_push_optuna_t0 | bert-base-uncased | student | last | 0.6010 | 0.6479 | 0.6865 | 0.7029 |
| r2_bert_uniform_push_optuna_t0 | bert-base-uncased | teacher | first_last | 0.6406 | 0.6974 | 0.7287 | 0.7421 |
| r2_bert_uniform_push_optuna_t0 | bert-base-uncased | teacher | last | 0.6009 | 0.6476 | 0.6862 | 0.7032 |
| r2_bert_uniform_push_optuna_t1 | bert-base-uncased | student | first_last | 0.6442 | 0.6985 | 0.7292 | 0.7421 |
| r2_bert_uniform_push_optuna_t1 | bert-base-uncased | student | last | 0.6061 | 0.6511 | 0.6885 | 0.7048 |
| r2_bert_uniform_push_optuna_t1 | bert-base-uncased | teacher | first_last | 0.6428 | 0.6976 | 0.7285 | 0.7419 |
| r2_bert_uniform_push_optuna_t1 | bert-base-uncased | teacher | last | 0.6040 | 0.6486 | 0.6866 | 0.7038 |
| r2_bert_uniform_push_optuna_t10 | bert-base-uncased | student | first_last | 0.6419 | 0.7002 | 0.7302 | 0.7427 |
| r2_bert_uniform_push_optuna_t10 | bert-base-uncased | student | last | 0.6053 | 0.6522 | 0.6882 | 0.7046 |
| r2_bert_uniform_push_optuna_t10 | bert-base-uncased | teacher | first_last | 0.6408 | 0.6991 | 0.7291 | 0.7427 |
| r2_bert_uniform_push_optuna_t10 | bert-base-uncased | teacher | last | 0.6028 | 0.6494 | 0.6864 | 0.7035 |
| r2_bert_uniform_push_optuna_t11 | bert-base-uncased | student | first_last | 0.6411 | 0.6962 | 0.7271 | 0.7405 |
| r2_bert_uniform_push_optuna_t11 | bert-base-uncased | student | last | 0.6016 | 0.6474 | 0.6850 | 0.7020 |
| r2_bert_uniform_push_optuna_t11 | bert-base-uncased | teacher | first_last | 0.6414 | 0.6972 | 0.7282 | 0.7416 |
| r2_bert_uniform_push_optuna_t11 | bert-base-uncased | teacher | last | 0.6024 | 0.6481 | 0.6862 | 0.7035 |
| r2_bert_uniform_push_optuna_t12 | bert-base-uncased | student | first_last | 0.6424 | 0.6982 | 0.7287 | 0.7417 |
| r2_bert_uniform_push_optuna_t12 | bert-base-uncased | student | last | 0.6052 | 0.6505 | 0.6874 | 0.7036 |
| r2_bert_uniform_push_optuna_t12 | bert-base-uncased | teacher | first_last | 0.6423 | 0.6980 | 0.7289 | 0.7422 |
| r2_bert_uniform_push_optuna_t12 | bert-base-uncased | teacher | last | 0.6044 | 0.6495 | 0.6869 | 0.7037 |
| r2_bert_uniform_push_optuna_t13 | bert-base-uncased | student | first_last | 0.6397 | 0.6996 | 0.7299 | 0.7427 |
| r2_bert_uniform_push_optuna_t13 | bert-base-uncased | student | last | 0.6030 | 0.6524 | 0.6885 | 0.7051 |
| r2_bert_uniform_push_optuna_t13 | bert-base-uncased | teacher | first_last | 0.6398 | 0.6992 | 0.7299 | 0.7434 |
| r2_bert_uniform_push_optuna_t13 | bert-base-uncased | teacher | last | 0.6025 | 0.6510 | 0.6881 | 0.7052 |
| r2_bert_uniform_push_optuna_t14 | bert-base-uncased | student | first_last | 0.6398 | 0.6973 | 0.7293 | 0.7422 |
| r2_bert_uniform_push_optuna_t14 | bert-base-uncased | student | last | 0.6016 | 0.6492 | 0.6878 | 0.7044 |
| r2_bert_uniform_push_optuna_t14 | bert-base-uncased | teacher | first_last | 0.6399 | 0.6972 | 0.7293 | 0.7426 |
| r2_bert_uniform_push_optuna_t14 | bert-base-uncased | teacher | last | 0.6014 | 0.6483 | 0.6872 | 0.7040 |
| r2_bert_uniform_push_optuna_t15 | bert-base-uncased | student | first_last | 0.6400 | 0.6989 | 0.7292 | 0.7422 |
| r2_bert_uniform_push_optuna_t15 | bert-base-uncased | student | last | 0.6015 | 0.6499 | 0.6877 | 0.7038 |
| r2_bert_uniform_push_optuna_t15 | bert-base-uncased | teacher | first_last | 0.6396 | 0.6983 | 0.7291 | 0.7424 |
| r2_bert_uniform_push_optuna_t15 | bert-base-uncased | teacher | last | 0.6005 | 0.6479 | 0.6863 | 0.7030 |
| r2_bert_uniform_push_optuna_t2 | bert-base-uncased | student | first_last | 0.6369 | 0.6958 | 0.7247 | 0.7391 |
| r2_bert_uniform_push_optuna_t2 | bert-base-uncased | student | last | 0.5978 | 0.6451 | 0.6805 | 0.6980 |
| r2_bert_uniform_push_optuna_t2 | bert-base-uncased | teacher | first_last | 0.6389 | 0.6970 | 0.7272 | 0.7412 |
| r2_bert_uniform_push_optuna_t2 | bert-base-uncased | teacher | last | 0.6004 | 0.6473 | 0.6837 | 0.7015 |
| r2_bert_uniform_push_optuna_t3 | bert-base-uncased | student | first_last | 0.6409 | 0.6985 | 0.7288 | 0.7417 |
| r2_bert_uniform_push_optuna_t3 | bert-base-uncased | student | last | 0.6015 | 0.6483 | 0.6860 | 0.7024 |
| r2_bert_uniform_push_optuna_t3 | bert-base-uncased | teacher | first_last | 0.6398 | 0.6974 | 0.7283 | 0.7416 |
| r2_bert_uniform_push_optuna_t3 | bert-base-uncased | teacher | last | 0.5997 | 0.6464 | 0.6844 | 0.7013 |
| r2_bert_uniform_push_optuna_t4 | bert-base-uncased | student | first_last | 0.6409 | 0.6986 | 0.7294 | 0.7423 |
| r2_bert_uniform_push_optuna_t4 | bert-base-uncased | student | last | 0.6029 | 0.6499 | 0.6876 | 0.7040 |
| r2_bert_uniform_push_optuna_t4 | bert-base-uncased | teacher | first_last | 0.6401 | 0.6979 | 0.7291 | 0.7424 |
| r2_bert_uniform_push_optuna_t4 | bert-base-uncased | teacher | last | 0.6013 | 0.6482 | 0.6865 | 0.7032 |
| r2_bert_uniform_push_optuna_t5 | bert-base-uncased | student | first_last | 0.6420 | 0.6993 | 0.7299 | 0.7425 |
| r2_bert_uniform_push_optuna_t5 | bert-base-uncased | student | last | 0.6055 | 0.6524 | 0.6893 | 0.7052 |
| r2_bert_uniform_push_optuna_t5 | bert-base-uncased | teacher | first_last | 0.6409 | 0.6985 | 0.7295 | 0.7426 |
| r2_bert_uniform_push_optuna_t5 | bert-base-uncased | teacher | last | 0.6036 | 0.6503 | 0.6877 | 0.7045 |
| r2_bert_uniform_push_optuna_t6 | bert-base-uncased | student | first_last | 0.6437 | 0.7012 | 0.7312 | 0.7436 |
| r2_bert_uniform_push_optuna_t6 | bert-base-uncased | student | last | 0.6084 | 0.6551 | 0.6904 | 0.7063 |
| r2_bert_uniform_push_optuna_t6 | bert-base-uncased | teacher | first_last | 0.6430 | 0.7003 | 0.7305 | 0.7435 |
| r2_bert_uniform_push_optuna_t6 | bert-base-uncased | teacher | last | 0.6066 | 0.6526 | 0.6888 | 0.7051 |
| r2_bert_uniform_push_optuna_t7 | bert-base-uncased | student | first_last | 0.6421 | 0.6992 | 0.7301 | 0.7426 |
| r2_bert_uniform_push_optuna_t7 | bert-base-uncased | student | last | 0.6043 | 0.6505 | 0.6881 | 0.7041 |
| r2_bert_uniform_push_optuna_t7 | bert-base-uncased | teacher | first_last | 0.6408 | 0.6980 | 0.7294 | 0.7424 |
| r2_bert_uniform_push_optuna_t7 | bert-base-uncased | teacher | last | 0.6020 | 0.6480 | 0.6863 | 0.7030 |
| r2_bert_uniform_push_optuna_t8 | bert-base-uncased | student | first_last | 0.6351 | 0.6931 | 0.7215 | 0.7362 |
| r2_bert_uniform_push_optuna_t8 | bert-base-uncased | student | last | 0.5931 | 0.6400 | 0.6752 | 0.6930 |
| r2_bert_uniform_push_optuna_t8 | bert-base-uncased | teacher | first_last | 0.6373 | 0.6955 | 0.7253 | 0.7398 |
| r2_bert_uniform_push_optuna_t8 | bert-base-uncased | teacher | last | 0.5969 | 0.6441 | 0.6809 | 0.6986 |
| r2_bert_uniform_push_optuna_t9 | bert-base-uncased | student | first_last | 0.6491 | 0.7005 | 0.7288 | 0.7415 |
| r2_bert_uniform_push_optuna_t9 | bert-base-uncased | student | last | 0.6098 | 0.6518 | 0.6862 | 0.7022 |
| r2_bert_uniform_push_optuna_t9 | bert-base-uncased | teacher | first_last | 0.6474 | 0.6999 | 0.7282 | 0.7414 |
| r2_bert_uniform_push_optuna_t9 | bert-base-uncased | teacher | last | 0.6073 | 0.6499 | 0.6845 | 0.7010 |
| r2_bert_uniform_push_seed43 | bert-base-uncased | student | first_last | 0.6297 | 0.6906 | 0.7170 | 0.7370 |
| r2_bert_uniform_push_seed43 | bert-base-uncased | student | last | 0.5900 | 0.6383 | 0.6678 | 0.6912 |
| r2_bert_uniform_push_seed43 | bert-base-uncased | teacher | first_last | 0.6352 | 0.6951 | 0.7240 | 0.7402 |
| r2_bert_uniform_push_seed43 | bert-base-uncased | teacher | last | 0.5973 | 0.6457 | 0.6784 | 0.6986 |
| r2_bert_uniform_push_tuned_s42 | bert-base-uncased | student | first_last | 0.6442 | 0.7016 | 0.7319 | 0.7430 |
| r2_bert_uniform_push_tuned_s42 | bert-base-uncased | student | last | 0.6103 | 0.6573 | 0.6915 | 0.7065 |
| r2_bert_uniform_push_tuned_s42 | bert-base-uncased | teacher | first_last | 0.6439 | 0.7009 | 0.7313 | 0.7429 |
| r2_bert_uniform_push_tuned_s42 | bert-base-uncased | teacher | last | 0.6092 | 0.6558 | 0.6903 | 0.7059 |
| r2_bert_uniform_push_tuned_s43 | bert-base-uncased | student | first_last | 0.6460 | 0.7028 | 0.7340 | 0.7457 |
| r2_bert_uniform_push_tuned_s43 | bert-base-uncased | student | last | 0.6118 | 0.6594 | 0.6951 | 0.7114 |
| r2_bert_uniform_push_tuned_s43 | bert-base-uncased | teacher | first_last | 0.6460 | 0.7023 | 0.7333 | 0.7456 |
| r2_bert_uniform_push_tuned_s43 | bert-base-uncased | teacher | last | 0.6111 | 0.6581 | 0.6940 | 0.7105 |
| r2_diag | answerdotai/ModernBERT-base | student | first_last | 0.6046 | 0.6642 | 0.6747 | 0.6906 |
| r2_diag | answerdotai/ModernBERT-base | student | last | 0.5780 | 0.6377 | 0.6563 | 0.6696 |
| r2_diag | answerdotai/ModernBERT-base | teacher | first_last | 0.6321 | 0.6857 | 0.6884 | 0.7075 |
| r2_diag | answerdotai/ModernBERT-base | teacher | last | 0.6114 | 0.6660 | 0.6742 | 0.6915 |
| r2_uniform_push | answerdotai/ModernBERT-base | student | first_last | 0.6366 | 0.6781 | 0.6864 | 0.7093 |
| r2_uniform_push | answerdotai/ModernBERT-base | student | last | 0.6137 | 0.6575 | 0.6725 | 0.6930 |
| r2_uniform_push | answerdotai/ModernBERT-base | teacher | first_last | 0.6587 | 0.6937 | 0.6961 | 0.7205 |
| r2_uniform_push | answerdotai/ModernBERT-base | teacher | last | 0.6409 | 0.6777 | 0.6858 | 0.7077 |
| r2_uniform_push_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6409 | 0.6786 | 0.6839 | 0.7101 |
| r2_uniform_push_seed43 | answerdotai/ModernBERT-base | student | last | 0.6164 | 0.6569 | 0.6688 | 0.6944 |
| r2_uniform_push_seed43 | answerdotai/ModernBERT-base | teacher | first_last | 0.6613 | 0.6941 | 0.6919 | 0.7202 |
| r2_uniform_push_seed43 | answerdotai/ModernBERT-base | teacher | last | 0.6425 | 0.6770 | 0.6805 | 0.7077 |
| r3_consistency_curriculum | answerdotai/ModernBERT-base | student | first_last | 0.6182 | 0.6606 | 0.6721 | 0.6988 |
| r3_consistency_curriculum | answerdotai/ModernBERT-base | student | last | 0.5925 | 0.6360 | 0.6557 | 0.6815 |
| r3_consistency_curriculum_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6082 | 0.6509 | 0.6750 | 0.7019 |
| r3_consistency_curriculum_seed43 | answerdotai/ModernBERT-base | student | last | 0.5768 | 0.6212 | 0.6550 | 0.6822 |
| r4_anchor_curriculum_velocity | answerdotai/ModernBERT-base | student | first_last | 0.6344 | 0.6636 | 0.6719 | 0.6968 |
| r4_anchor_curriculum_velocity | answerdotai/ModernBERT-base | student | last | 0.6112 | 0.6400 | 0.6568 | 0.6800 |
| r4_anchor_curriculum_velocity_ext3000 | answerdotai/ModernBERT-base | student | first_last | 0.5707 | 0.6072 | 0.6205 | 0.6418 |
| r4_anchor_curriculum_velocity_ext3000 | answerdotai/ModernBERT-base | student | last | 0.5309 | 0.5670 | 0.5892 | 0.5989 |
| r4_anchor_curriculum_velocity_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6306 | 0.6581 | 0.6722 | 0.7058 |
| r4_anchor_curriculum_velocity_seed43 | answerdotai/ModernBERT-base | student | last | 0.6024 | 0.6313 | 0.6547 | 0.6887 |
| r5a_temp_schedule | answerdotai/ModernBERT-base | student | first_last | 0.7045 | 0.7248 | 0.7241 | 0.7411 |
| r5a_temp_schedule | answerdotai/ModernBERT-base | student | last | 0.6931 | 0.7147 | 0.7139 | 0.7329 |
| r5a_temp_schedule_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6989 | 0.7269 | 0.7414 | 0.7461 |
| r5a_temp_schedule_seed43 | answerdotai/ModernBERT-base | student | last | 0.6861 | 0.7186 | 0.7329 | 0.7396 |
| r5b_momentum_ramp | answerdotai/ModernBERT-base | student | first_last | 0.6943 | 0.7130 | 0.7153 | 0.7361 |
| r5b_momentum_ramp | answerdotai/ModernBERT-base | student | last | 0.6837 | 0.7012 | 0.7068 | 0.7280 |
| r5b_momentum_ramp_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6820 | 0.7087 | 0.7036 | 0.7337 |
| r5b_momentum_ramp_seed43 | answerdotai/ModernBERT-base | student | last | 0.6664 | 0.6947 | 0.6939 | 0.7233 |
| r5c_fixed_high_t | answerdotai/ModernBERT-base | student | first_last | 0.6188 | 0.6685 | 0.6771 | 0.7001 |
| r5c_fixed_high_t | answerdotai/ModernBERT-base | student | last | 0.5974 | 0.6464 | 0.6624 | 0.6825 |
| r5c_fixed_high_t_ext3000 | answerdotai/ModernBERT-base | student | first_last | 0.5008 | 0.5585 | 0.5886 | 0.5843 |
| r5c_fixed_high_t_ext3000 | answerdotai/ModernBERT-base | student | last | 0.4611 | 0.5160 | 0.5513 | 0.5510 |
| r5c_fixed_high_t_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.6113 | 0.6512 | 0.6754 | 0.7031 |
| r5c_fixed_high_t_seed43 | answerdotai/ModernBERT-base | student | last | 0.5828 | 0.6216 | 0.6589 | 0.6869 |
| r5d_bert_combined | bert-base-uncased | student | first_last | 0.6405 | 0.7061 | 0.7354 | 0.7449 |
| r5d_bert_combined | bert-base-uncased | student | last | 0.6119 | 0.6676 | 0.6985 | 0.7124 |
| r5d_bert_combined_seed43 | bert-base-uncased | student | first_last | 0.6416 | 0.7047 | 0.7347 | 0.7447 |
| r5d_bert_combined_seed43 | bert-base-uncased | student | last | 0.6106 | 0.6652 | 0.6959 | 0.7111 |
| r5d_bert_embed_uniform_push_s42 | bert-base-uncased | student | first_last | 0.6406 | 0.7068 | 0.7353 | 0.7446 |
| r5d_bert_embed_uniform_push_s42 | bert-base-uncased | student | last | 0.6128 | 0.6690 | 0.6986 | 0.7123 |
| r5d_bert_embed_uniform_push_s42 | bert-base-uncased | teacher | first_last | 0.6417 | 0.7057 | 0.7336 | 0.7441 |
| r5d_bert_embed_uniform_push_s42 | bert-base-uncased | teacher | last | 0.6112 | 0.6640 | 0.6941 | 0.7092 |
| r5d_bert_embed_uniform_push_s43 | bert-base-uncased | student | first_last | 0.6401 | 0.7032 | 0.7331 | 0.7432 |
| r5d_bert_embed_uniform_push_s43 | bert-base-uncased | student | last | 0.6083 | 0.6628 | 0.6934 | 0.7085 |
| r5d_bert_embed_uniform_push_s43 | bert-base-uncased | teacher | first_last | 0.6394 | 0.7019 | 0.7313 | 0.7423 |
| r5d_bert_embed_uniform_push_s43 | bert-base-uncased | teacher | last | 0.6051 | 0.6585 | 0.6898 | 0.7056 |
| r5d_bert_uniform_push_tuned_s42 | bert-base-uncased | student | first_last | 0.6419 | 0.7065 | 0.7353 | 0.7448 |
| r5d_bert_uniform_push_tuned_s42 | bert-base-uncased | student | last | 0.6123 | 0.6670 | 0.6973 | 0.7117 |
| r5d_bert_uniform_push_tuned_s42 | bert-base-uncased | teacher | first_last | 0.6413 | 0.7047 | 0.7332 | 0.7438 |
| r5d_bert_uniform_push_tuned_s42 | bert-base-uncased | teacher | last | 0.6093 | 0.6614 | 0.6925 | 0.7078 |
| r5d_bert_uniform_push_tuned_s43 | bert-base-uncased | student | first_last | 0.6402 | 0.7038 | 0.7337 | 0.7438 |
| r5d_bert_uniform_push_tuned_s43 | bert-base-uncased | student | last | 0.6074 | 0.6634 | 0.6938 | 0.7092 |
| r5d_bert_uniform_push_tuned_s43 | bert-base-uncased | teacher | first_last | 0.6399 | 0.7021 | 0.7311 | 0.7422 |
| r5d_bert_uniform_push_tuned_s43 | bert-base-uncased | teacher | last | 0.6050 | 0.6580 | 0.6890 | 0.7052 |
| r5d_combined | answerdotai/ModernBERT-base | student | first_last | 0.7239 | 0.7388 | 0.7444 | 0.7566 |
| r5d_combined | answerdotai/ModernBERT-base | student | last | 0.7151 | 0.7316 | 0.7359 | 0.7505 |
| r5d_combined_ext3000 | answerdotai/ModernBERT-base | student | first_last | 0.6872 | 0.7106 | 0.7142 | 0.7356 |
| r5d_combined_ext3000 | answerdotai/ModernBERT-base | student | last | 0.6786 | 0.6999 | 0.7067 | 0.7283 |
| r5d_combined_ext5000 | answerdotai/ModernBERT-base | student | first_last | 0.5777 | 0.6330 | 0.6605 | 0.6665 |
| r5d_combined_ext5000 | answerdotai/ModernBERT-base | student | last | 0.5450 | 0.5955 | 0.6373 | 0.6331 |
| r5d_combined_seed43 | answerdotai/ModernBERT-base | student | first_last | 0.7107 | 0.7342 | 0.7397 | 0.7552 |
| r5d_combined_seed43 | answerdotai/ModernBERT-base | student | last | 0.7020 | 0.7276 | 0.7258 | 0.7511 |
| r5d_diag | answerdotai/ModernBERT-base | student | first_last | 0.7239 | 0.7388 | 0.7444 | 0.7566 |
| r5d_diag | answerdotai/ModernBERT-base | student | last | 0.7151 | 0.7316 | 0.7359 | 0.7505 |
| r5d_diag | answerdotai/ModernBERT-base | teacher | first_last | 0.7276 | 0.7437 | 0.7503 | 0.7604 |
| r5d_diag | answerdotai/ModernBERT-base | teacher | last | 0.7185 | 0.7370 | 0.7420 | 0.7553 |

## backbone별 후처리 평균 효과 (student만, vs postprocess=none, Δspearman)
| backbone | center | center_pc1 | center_pc2 |
|---|---|---|---|
| answerdotai/ModernBERT-base | +0.0390 | +0.0550 | +0.0748 |
| bert-base-uncased | +0.0512 | +0.0817 | +0.0949 |

## student vs teacher (teacher_state_dict 있는 run만, pooling=last/postprocess=none 기준)
| run | student | teacher | teacher - student |
|---|---|---|---|
| r2_bert_uniform_push | 0.5784 | 0.5912 | +0.0128 |
| r2_bert_uniform_push_optuna_t0 | 0.6010 | 0.6009 | -0.0001 |
| r2_bert_uniform_push_optuna_t1 | 0.6061 | 0.6040 | -0.0021 |
| r2_bert_uniform_push_optuna_t10 | 0.6053 | 0.6028 | -0.0025 |
| r2_bert_uniform_push_optuna_t11 | 0.6016 | 0.6024 | +0.0007 |
| r2_bert_uniform_push_optuna_t12 | 0.6052 | 0.6044 | -0.0008 |
| r2_bert_uniform_push_optuna_t13 | 0.6030 | 0.6025 | -0.0005 |
| r2_bert_uniform_push_optuna_t14 | 0.6016 | 0.6014 | -0.0003 |
| r2_bert_uniform_push_optuna_t15 | 0.6015 | 0.6005 | -0.0011 |
| r2_bert_uniform_push_optuna_t2 | 0.5978 | 0.6004 | +0.0026 |
| r2_bert_uniform_push_optuna_t3 | 0.6015 | 0.5997 | -0.0019 |
| r2_bert_uniform_push_optuna_t4 | 0.6029 | 0.6013 | -0.0016 |
| r2_bert_uniform_push_optuna_t5 | 0.6055 | 0.6036 | -0.0019 |
| r2_bert_uniform_push_optuna_t6 | 0.6084 | 0.6066 | -0.0018 |
| r2_bert_uniform_push_optuna_t7 | 0.6043 | 0.6020 | -0.0023 |
| r2_bert_uniform_push_optuna_t8 | 0.5931 | 0.5969 | +0.0038 |
| r2_bert_uniform_push_optuna_t9 | 0.6098 | 0.6073 | -0.0025 |
| r2_bert_uniform_push_seed43 | 0.5900 | 0.5973 | +0.0073 |
| r2_bert_uniform_push_tuned_s42 | 0.6103 | 0.6092 | -0.0011 |
| r2_bert_uniform_push_tuned_s43 | 0.6118 | 0.6111 | -0.0007 |
| r2_diag | 0.5780 | 0.6114 | +0.0335 |
| r2_uniform_push | 0.6137 | 0.6409 | +0.0273 |
| r2_uniform_push_seed43 | 0.6164 | 0.6425 | +0.0261 |
| r5d_bert_embed_uniform_push_s42 | 0.6128 | 0.6112 | -0.0016 |
| r5d_bert_embed_uniform_push_s43 | 0.6083 | 0.6051 | -0.0032 |
| r5d_bert_uniform_push_tuned_s42 | 0.6123 | 0.6093 | -0.0030 |
| r5d_bert_uniform_push_tuned_s43 | 0.6074 | 0.6050 | -0.0023 |
| r5d_diag | 0.7151 | 0.7185 | +0.0034 |


## 7-task 평균 (best combo: pooling=first_last, postprocess=center_pc2)

| run | entity | avg_7task (raw, postprocess=none) | avg_7task (best combo) | 개선 |
|---|---|---|---|---|
| r5d_bert_combined | student | 0.5582 | 0.6617 | +0.1035 |
| r5d_bert_lrsplit_b_s43 | student | 0.5868 | 0.6710 | +0.0842 |
| r5d_bert_lrsplit_b_s43 | teacher | 0.5808 | 0.6691 | +0.0883 |
