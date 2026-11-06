"""
Étape 1 : chargement des images et détection des coins du damier.
"""

from pathlib import Path

import cv2
import numpy as np

import json

# Critère d'arrêt de l'affinage sous-pixellique des coins :
# on s'arrête après 30 itérations OU quand le coin bouge de moins de 0.001 pixel.
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def list_images(folder, extensions=(".jpg", ".jpeg", ".png")):
    """Renvoie la liste triée des chemins d'images d'un dossier."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)


def build_object_points(pattern_size, square_size_mm):
    """
    Coordonnées 3D des coins intérieurs du damier dans le repère MONDE de la mire.

    Le repère monde est posé sur la mire : origine au premier coin, axes X et Y
    le long des bords des cases, et Z = 0 pour tous les points (la mire est plane).
    C'est l'équivalent du "fichier config" X, Y, Z dont parle le cours.

    pattern_size   : (nb_coins_par_ligne, nb_coins_par_colonne), coins INTÉRIEURS
    square_size_mm : côté d'une case, mesuré à la règle sur la feuille imprimée
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    # Grille (0,0), (1,0), (2,0), ... dans le même ordre que celui renvoyé
    # par findChessboardCorners (ligne par ligne).
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_size_mm


def detect_corners(image_paths, pattern_size, max_side=1600, verbose=True):
    """
    Détecte les coins intérieurs du damier dans chaque image.

    Pour aller vite sur des photos de téléphone (12 Mpx et plus), la détection
    grossière se fait sur une version réduite de l'image (côté max = max_side).
    Les coins sont ensuite remis à l'échelle et affinés au sous-pixel sur
    l'image en PLEINE résolution : on ne perd donc pas en précision.

    Renvoie :
      detections : liste de dicts {"path", "found", "corners"}
      image_size : (largeur, hauteur) commune à toutes les images retenues
    """
    detections = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))  # applique l'orientation EXIF par défaut
        if img is None:
            print(f"[!] Impossible de lire {path.name}")
            continue

        h, w = img.shape[:2]
        # Toutes les images doivent avoir la même taille (et la même orientation),
        # sinon la calibration n'a pas de sens : K dépend de la taille de l'image.
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            print(f"[!] {path.name} : taille {w}x{h} différente de {image_size} "
                  f"-> image ignorée (photo prise en portrait ?)")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) Détection grossière sur image réduite
        scale = min(1.0, max_side / max(w, h))
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1 else gray
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(small, pattern_size, flags)

        # 2) Retour en pleine résolution + affinage sous-pixellique
        if found:
            corners = (corners / scale).astype(np.float32)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       SUBPIX_CRITERIA)

        detections.append({"path": path, "found": found,
                           "corners": corners if found else None})
        if verbose:
            print(f"{path.name:>12} : {'OK' if found else 'ÉCHEC'}")

    n_ok = sum(d["found"] for d in detections)
    if verbose:
        print(f"\nDamier détecté dans {n_ok}/{len(detections)} images "
              f"(taille d'image : {image_size[0]}x{image_size[1]} px)")
    return detections, image_size


def save_detection_images(detections, pattern_size, out_dir, max_side=1200):
    """
    Dessine les coins détectés sur chaque image et sauvegarde le résultat
    (réduit à max_side pixels pour rester léger dans le rapport).
    Permet de VÉRIFIER visuellement que la détection est correcte.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in detections:
        if not d["found"]:
            continue
        img = cv2.imread(str(d["path"]))
        cv2.drawChessboardCorners(img, pattern_size, d["corners"], True)
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(w, h))
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"corners_{d['path'].stem}.jpg"), img)



# Étape 2 : calibration et erreur de reprojection




def calibrate(detections, pattern_size, square_size_mm, image_size, flags=0):
    """
    Calibre la caméra (méthode de Zhang implémentée par OpenCV).

    Pour chaque image où le damier a été trouvé, on associe :
      - les points 3D de la mire (repère monde, Z = 0, en mm)
      - les points 2D détectés (en pixels)
    calibrateCamera estime alors K, la distorsion, et la pose (R, t) de chaque vue.
    """
    objp = build_object_points(pattern_size, square_size_mm)
    kept = [d for d in detections if d["found"]]

    object_points = [objp for _ in kept]        # mêmes points 3D pour chaque vue
    image_points = [d["corners"] for d in kept]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )

    return {
        "rms": rms,                 # erreur de reprojection RMS globale (px)
        "K": K,                     # matrice intrinsèque 3x3
        "dist": dist.ravel(),       # (k1, k2, p1, p2, k3)
        "rvecs": rvecs,             # rotation de chaque vue (vecteur de Rodrigues)
        "tvecs": tvecs,             # translation de chaque vue (en mm)
        "names": [d["path"].name for d in kept],
        "object_points": object_points,
        "image_points": image_points,
        "image_size": image_size,
    }


def per_image_errors(calib):
    """
    Erreur de reprojection RMS de CHAQUE image (en pixels) :
    on reprojette les points 3D avec les paramètres estimés et on compare
    aux coins détectés. Sert à repérer les images aberrantes.
    """
    errors = {}
    for name, objp, imgp, rvec, tvec in zip(
        calib["names"], calib["object_points"], calib["image_points"],
        calib["rvecs"], calib["tvecs"]
    ):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, calib["K"], calib["dist"])
        diff = imgp.reshape(-1, 2) - proj.reshape(-1, 2)
        errors[name] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    return errors


def print_calibration(calib):
    """Affiche les résultats de manière lisible."""
    K, (k1, k2, p1, p2, k3) = calib["K"], calib["dist"]
    w, h = calib["image_size"]
    print(f"Nombre d'images utilisées : {len(calib['names'])}")
    print(f"Erreur RMS globale        : {calib['rms']:.3f} px\n")
    print("Matrice K :\n", np.array2string(K, precision=2, suppress_small=True))
    print(f"\nfx = {K[0,0]:.1f} px   fy = {K[1,1]:.1f} px   fy/fx = {K[1,1]/K[0,0]:.4f}")
    print(f"cx = {K[0,2]:.1f} px   (centre image : {w/2:.1f}, écart {K[0,2]-w/2:+.1f})")
    print(f"cy = {K[1,2]:.1f} px   (centre image : {h/2:.1f}, écart {K[1,2]-h/2:+.1f})")
    print(f"\nDistorsion : k1={k1:.4f}  k2={k2:.4f}  p1={p1:.5f}  p2={p2:.5f}  k3={k3:.4f}")


def save_calibration(calib, path):
    """Sauvegarde K et la distorsion (JSON) pour les étapes suivantes."""
    data = {
        "rms": calib["rms"],
        "image_size": list(calib["image_size"]),
        "K": calib["K"].tolist(),
        "dist": calib["dist"].tolist(),
        "names": calib["names"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)