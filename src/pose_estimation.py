"""

Une fois la caméra calibrée (K et distorsion connus), on estime pour chaque
nouvelle image la POSE de la caméra par rapport au damier (R, t) :
c'est le problème PnP (Perspective-n-Point) étudié dans l'article de Quan et Lan.
On s'en sert ensuite pour projeter un cube 3D virtuel posé sur le damier.
"""

import cv2
import numpy as np

from src.calibration import build_object_points


def estimate_pose(corners, K, dist, pattern_size, square_size_mm):
    """
    Estime la pose de la caméra à partir des coins détectés d'UNE image.

    corners : coins détectés (pixels), dans l'ordre de findChessboardCorners
    Renvoie un dict avec rvec, tvec, R, l'erreur de reprojection de la pose,
    la distance caméra-damier et l'inclinaison du damier par rapport à l'axe optique.
    """
    objp = build_object_points(pattern_size, square_size_mm)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)

    # Qualité de la pose : on reprojette les points 3D et on compare aux coins détectés
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    diff = corners.reshape(-1, 2) - proj.reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    # R : matrice de rotation 3x3 (rvec est sa forme compacte, le vecteur de Rodrigues)
    R, _ = cv2.Rodrigues(rvec)
    # La normale au damier, exprimée dans le repère caméra, est la 3e colonne de R.
    # Son angle avec l'axe optique (Z caméra) donne l'inclinaison du damier.
    tilt = float(np.degrees(np.arccos(abs(R[2, 2]))))

    return {
        "ok": ok,
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "err": err,                                   # px
        "dist_mm": float(np.linalg.norm(tvec)),       # distance au 1er coin du damier
        "tilt_deg": tilt,                             # 0° = damier vu de face
    }


def draw_cube(img, pose, K, dist, square_size_mm, size_squares=2):
    """
    Dessine un cube virtuel posé sur le damier, ainsi que les axes du repère monde.
    Le cube a pour côté size_squares cases.

    Remarque : si le cube s'enfonce dans la table au lieu de se dresser au-dessus,
    il suffit d'inverser le signe de la coordonnée Z (-s -> +s).
    """
    s = size_squares * square_size_mm
    cube = np.float32([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],        # base, sur le damier
        [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s],    # face supérieure
    ])
    pts, _ = cv2.projectPoints(cube, pose["rvec"], pose["tvec"], K, dist)
    pts = [tuple(map(int, p)) for p in pts.reshape(-1, 2)]

    out = img.copy()
    t = max(3, img.shape[1] // 300)                          # épaisseur adaptée à la taille

    cv2.fillPoly(out, [np.array(pts[:4])], (0, 200, 0))      # base pleine (verte)
    for i in range(4):
        cv2.line(out, pts[i], pts[i + 4], (255, 0, 0), t)    # arêtes verticales (bleues)
    cv2.polylines(out, [np.array(pts[4:])], True, (0, 0, 255), t)  # dessus (rouge)

    # Axes du repère monde : X rouge, Y vert, Z bleu
    cv2.drawFrameAxes(out, K, dist, pose["rvec"], pose["tvec"], 3 * square_size_mm, t)
    return out