import riepybdlib as rbd

Manifold_T = rbd.manifold.get_euclidean_manifold(1, "TIME")
Manifold_R1 = rbd.manifold.get_euclidean_manifold(1, "R1")
Manifold_R3 = rbd.manifold.get_euclidean_manifold(3, "R3")
Manifold_S1 = rbd.manifold.get_s1_manifold()
Manifold_S2 = rbd.manifold.get_s2hat_manifold()
# Manifold_S2 = rbd.manifold.get_s2_manifold()
Manifold_Quat = rbd.manifold.get_quaternion_manifold("QUAT")
