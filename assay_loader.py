import dill


with open("exp01.pkl", "rb") as f:
    assay = dill.load(f)

