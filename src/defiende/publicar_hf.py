"""Publica hf/<version>/ como un commit del repo de modelos, lo etiqueta y registra el SHA completo.

Cada revisión contiene SOLO los archivos de su versión (delete_patterns="*"), de modo que descargar
el SHA de v1 o de v2 entrega esa solución completa.

Uso: python -m defiende.publicar_hf --version v1 --repo USUARIO/REPO [--privado]
"""

from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi

from .comun import EVIDENCIAS, RAIZ, guardar_json, sha256_archivo


def actualizar_card(version: str, repo: str) -> None:
    """Revisión de solo documentación; verifica que model.joblib sigue siendo el artefacto congelado."""
    carpeta = RAIZ / "hf" / version
    config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    api = HfApi()
    info = api.upload_file(path_or_fileobj=str(carpeta / "README.md"), path_in_repo="README.md", repo_id=repo,
                           commit_message=f"{version}: model card con la evaluación final en test (modelo sin cambios)")
    archivos = {f.path: f for f in api.list_repo_tree(repo, revision=info.oid)}
    if archivos["model.joblib"].lfs.sha256 != config["model_sha256"]:
        raise SystemExit("model.joblib cambió: la revisión de documentación no es válida")
    ruta = EVIDENCIAS / "versiones.json"
    doc = json.loads(ruta.read_text(encoding="utf-8"))
    doc[version]["huggingface"]["revision_documentacion"] = {
        "sha": info.oid, "cambio": "solo README.md (resultado de test)", "model_sha256_verificado": config["model_sha256"]}
    guardar_json(doc, ruta)
    print(f"model card actualizada: {repo}@{info.oid}; model.joblib intacto ({config['model_sha256'][:12]}…)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--privado", action="store_true")
    ap.add_argument("--solo-card", action="store_true", help="sube solo README.md (no cambia el modelo ni los tags)")
    args = ap.parse_args()
    if args.solo_card:
        actualizar_card(args.version, args.repo)
        return

    carpeta = RAIZ / "hf" / args.version
    config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    if sha256_archivo(carpeta / "model.joblib") != config["model_sha256"]:
        raise SystemExit("model.joblib no coincide con config.json")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.privado, exist_ok=True)
    info = api.upload_folder(
        repo_id=args.repo,
        folder_path=str(carpeta),
        commit_message=f"{args.version}: {config['exp_id']} umbral {config['umbral']} (run {config['mlflow_run_id']})",
        delete_patterns="*",
    )
    sha = info.oid
    api.create_tag(args.repo, tag=args.version, revision=sha, tag_message=f"Solución {args.version}", exist_ok=False)

    ref = {
        "repo": args.repo,
        "url": f"https://huggingface.co/{args.repo}",
        "revision_sha": sha,
        "tag": args.version,
        "url_revision": f"https://huggingface.co/{args.repo}/tree/{sha}",
    }
    for ruta in (EVIDENCIAS / "versiones.json", EVIDENCIAS / "seleccion.json"):
        if ruta.exists():
            doc = json.loads(ruta.read_text(encoding="utf-8"))
            if args.version in doc:
                doc[args.version]["huggingface"] = ref
                guardar_json(doc, ruta)
    print(f"{args.version} publicado: {args.repo}@{sha}")


if __name__ == "__main__":
    main()
