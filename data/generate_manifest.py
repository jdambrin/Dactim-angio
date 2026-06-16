#!/usr/bin/env python3
"""
Génère un fichier manifest.json pour l'explorateur d'images patients.

Usage :
    python3 generate_manifest.py /chemin/vers/data/experiment3

Si aucun chemin n'est donné, le dossier courant est utilisé.

Le manifest est écrit DANS le dossier scanné, au format :
    { "P001": ["img_001.png", "img_002.png"], "P002": ["scan_01.jpg"] }

Les chemins listés sont relatifs au dossier de chaque patient (les
sous-dossiers internes sont conservés, ex. "serie1/img_001.png").
Les images à la racine du dossier scanné sont regroupées sous "(Racine)".
"""

import os
import re
import sys
import json

ACCEPTED = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def natural_key(s):
    # Tri "naturel" : img_2 avant img_10
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def collect_images(folder):
    """Retourne la liste des images d'un dossier, récursivement,
    en chemins relatifs à ce dossier (séparateur '/')."""
    images = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in ACCEPTED:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, folder).replace(os.sep, "/")
                images.append(rel)
    images.sort(key=natural_key)
    return images


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        print(f"Erreur : '{root}' n'est pas un dossier.", file=sys.stderr)
        sys.exit(1)

    manifest = {}

    # Images directement à la racine -> "(Racine)"
    root_imgs = [
        f for f in os.listdir(root)
        if os.path.isfile(os.path.join(root, f))
        and os.path.splitext(f)[1].lower() in ACCEPTED
    ]
    root_imgs.sort(key=natural_key)
    if root_imgs:
        manifest["(Racine)"] = root_imgs

    # Chaque sous-dossier de premier niveau = un patient
    for entry in sorted(os.listdir(root), key=natural_key):
        sub = os.path.join(root, entry)
        if os.path.isdir(sub):
            imgs = collect_images(sub)
            if imgs:
                manifest[entry] = imgs

    out_path = os.path.join(root, "manifest.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in manifest.values())
    print(f"manifest.json écrit dans : {out_path}")
    print(f"{len(manifest)} patient(s), {total} image(s) au total.")


if __name__ == "__main__":
    main()
