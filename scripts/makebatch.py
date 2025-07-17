import pathlib
import yaml

batch = {
    "CONFIG": {
        "ALIAS": {"t1map": "search", "refdata": "getref"},
        "PARAMETERS": {"t1map": {"shape": [256, 256]}},
    }
}

root = pathlib.Path("simulations")
simulations = [root / "DataBase_Num_Ph", root / "DataBase_Num_Ph_2"]

for base in simulations:
    for path in base.glob("V*/Phantom*"):
         id = ".".join(path.parts[-3:])
         batch[id] = {
             "t1map": {"path": str(path / "KSpaceData.mat")}, 
             "refdata": {"path": str(path / "ParamMap.mat")},
         }

with open("batch_phantoms.yml", "w") as fp:
    yaml.dump(batch, fp)
