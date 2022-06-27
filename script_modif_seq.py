
import json
import numpy as np
import matplotlib.pyplot as plt

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)

sequence_config.keys()
TE = sequence_config["TE"]
TR = sequence_config["TR"]
B1 = sequence_config["B1"]

unique_TE=np.unique(TE)
first_switch = np.argwhere(TE==unique_TE[1])[0][0]

plateau_end = 600

sequence_config["TE"] = TE[:plateau_end]+TE[first_switch:]
sequence_config["TR"] = TR[:plateau_end]+TR[first_switch:]
sequence_config["B1"] = B1[:plateau_end]+B1[first_switch:]

file_seq_modif="mrf_sequence_plateau{}.json".format(plateau_end)

with open(file_seq_modif,"w") as file:
    json.dump(sequence_config,file)

plt.figure()
plt.plot(sequence_config["TE"])

