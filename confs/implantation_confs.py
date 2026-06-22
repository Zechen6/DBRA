ATTACK_ALL = True

dictionary_top_rate = 0.02
victim_sample_top_rate = 0.02
rest_signal_rate = 1
implanted_signal_rate = 0.8
adv_client = 0
no_class_client = 1
if ATTACK_ALL:
    victim_client = list(range(0,20))
    victim_client.remove(adv_client)
    #victim_client.remove(no_class_client)
else:
    victim_client = 3
target_label = 0
trigger_search_ffc_params = {'lr':100, 'echo':10}
trigger_extraction_ffc_params = {'lr':100, 'echo':10}
max_attack_times = 5
