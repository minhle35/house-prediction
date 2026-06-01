"""API server entrypoint with explicit mode selection.

Usage:
    uv run serve --mode local --train-data ../data/train.csv
    uv run serve --mode local                     # uses cached models if already trained
    uv run serve --mode azure                     # loads from Azure Blob (needs .env)
    uv run serve --mode azure --port 8080 --reload
"""

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="House Prediction API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  local   Load from cached joblib files in MODEL_DIR.
          If no cache exists, train from --train-data CSV first.
          Azure Blob is never contacted.

  azure   Download models from Azure Blob Storage on startup.
          Requires AZURE_STORAGE_CONNECTION_STRING in .env or environment.
          Falls back to local cache if download fails.
        """,
    )
    cli.add_argument(
        "--mode",
        choices=["local", "azure"],
        required=True,
        help="local = CSV/joblib cache only, azure = Azure Blob primary",
    )
    cli.add_argument(
        "--train-data",
        metavar="PATH",
        help="(local mode) CSV path to train from when no cached models exist",
    )
    cli.add_argument(
        "--model-dir",
        metavar="DIR",
        default=None,
        help="Directory for cached joblib files (default: models/)",
    )
    cli.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    cli.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    cli.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (dev only)",
    )
    cli.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )
    return cli.parse_args()


def _configure_local(args: argparse.Namespace) -> None:
    """Set env vars for local mode and validate that models can be found or trained."""
    # Prevent accidental Azure contact in local mode
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ""

    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir

    model_dir = Path(os.environ.get("MODEL_DIR", "models"))
    cached = all(
        (model_dir / f"{name}.joblib").exists()
        for name in ("regression", "classification", "label_encoder")
    )

    if cached:
        print(f"[serve] local mode — loading cached models from {model_dir}/")
    elif args.train_data:
        train_path = Path(args.train_data)
        if not train_path.exists():
            print(f"[serve] error: --train-data file not found: {train_path}", file=sys.stderr)
            sys.exit(1)
        os.environ["TRAIN_DATA_PATH"] = str(train_path.resolve())
        print(f"[serve] local mode — no cache found, will train from {train_path}")
    else:
        print(
            "[serve] error: no cached models found in '{}' and --train-data not provided.\n"
            "        Run with --train-data PATH to train on first startup, or\n"
            "        use --mode azure to load from Azure Blob.".format(model_dir),
            file=sys.stderr,
        )
        sys.exit(1)


def _configure_azure(args: argparse.Namespace) -> None:
    """Validate that the Azure connection string is available."""
    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        print(
            "[serve] error: AZURE_STORAGE_CONNECTION_STRING is not set.\n"
            "        Add it to backend/.env or export it before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[serve] azure mode — models will be downloaded from Azure Blob on startup")


def main() -> None:
    args = _parse_args()

    if args.mode == "local":
        _configure_local(args)
    else:
        _configure_azure(args)

    log_level = ["warning", "info", "debug"][min(args.verbose, 2)]

    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
