

# rigidogram
"""
ca = u.select_atoms("protein and name CA")

for m in ca.trajectory:
    for n in ca.trajectory:
        eyes = []
        for i in m.residues:
            jays = []
            for j in n.residues:
                jays.append(np.linalg.norm(i.))
"""