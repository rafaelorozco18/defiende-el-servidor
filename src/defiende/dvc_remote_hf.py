"""Sube la caché DVC a un repo *dataset* de Hugging Face que actúa como remote HTTP de DVC.

DVC (>= 3) guarda cada objeto en `files/md5/<2 car.>/<resto>` tanto en la caché como en el remote.
Hugging Face no acepta el PUT de `dvc push`, así que el push se hace con huggingface_hub copiando esa
misma estructura; la lectura (`dvc pull`) usa el remote HTTP de .dvc/config sin credenciales.

Uso: python -m defiende.dvc_remote_hf --repo USUARIO/defiende-el-servidor-dvc
"""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi

from .comun import RAIZ, git


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    cache = RAIZ / ".dvc" / "cache" / "files" / "md5"
    objetos = [p for p in cache.rglob("*") if p.is_file()]
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    info = api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(cache),
        path_in_repo="files/md5",
        commit_message=f"Caché DVC del commit {git('rev-parse', 'HEAD')} ({len(objetos)} objetos)",
    )
    readme = (
        "---\nlicense: other\ntags: [dvc, unsw-nb15]\n---\n\n"
        "# Remote DVC — Defiende el servidor\n\n"
        "Almacenamiento DVC (estructura `files/md5/`) del repo de código "
        "https://github.com/rafaelorozco18/defiende-el-servidor . No se usa directamente: se lee con\n\n"
        "```bash\nuv run dvc pull\n```\n\n"
        "Contiene las particiones publicadas de UNSW-NB15 (Moustafa y Slay, 2015; "
        "https://research.unsw.edu.au/projects/unsw-nb15-dataset), sujetas a las condiciones de uso de UNSW, "
        "y los datos preparados derivados.\n"
    )
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=args.repo, repo_type="dataset",
                    commit_message="README del remote DVC")
    print(f"{len(objetos)} objetos subidos a {args.repo} (commit {info.oid})")


if __name__ == "__main__":
    main()
