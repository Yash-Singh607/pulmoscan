"""Unified command-line interface.

Usage::

    python -m chestxray.cli setup-data
    python -m chestxray.cli eda --data-dir data/chest_xray
    python -m chestxray.cli train --epochs 15 --batch-size 32
    python -m chestxray.cli predict --image path/to.jpg
    python -m chestxray.cli serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import os

from .config import Config
from .utils import configure_logging, get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chestxray", description="Chest X-Ray Pneumonia Classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    # setup-data
    p_setup = sub.add_parser("setup-data", help="Download the Kaggle dataset")
    p_setup.add_argument("--data-dir", "--data_dir", dest="data_dir", default="data/chest_xray")

    # eda
    p_eda = sub.add_parser("eda", help="Run exploratory data analysis")
    p_eda.add_argument("--data-dir", "--data_dir", dest="data_dir", default="data/chest_xray")
    p_eda.add_argument("--save-dir", "--save_dir", dest="save_dir", default="outputs/eda")

    # train
    p_train = sub.add_parser("train", help="Train the classifier")
    p_train.add_argument("--data-dir", "--data_dir", dest="data_dir", default="data/chest_xray")
    p_train.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument(
        "--unfreeze-epoch", "--unfreeze_epoch", dest="unfreeze_epoch", type=int, default=None
    )
    p_train.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=None
    )
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--no-class-weights", action="store_true")
    p_train.add_argument(
        "--label-smoothing", "--label_smoothing", dest="label_smoothing", type=float, default=None,
        help="Label smoothing factor for the loss (default 0.05)",
    )
    p_train.add_argument(
        "--val-split", "--val_split", dest="val_split", type=float, default=None,
        help="Fraction of training data held out for validation (default 0.15)",
    )
    p_train.add_argument(
        "--no-resplit-val", dest="no_resplit_val", action="store_true",
        help="Use the dataset's tiny shipped val folder instead of re-splitting",
    )
    p_train.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap images per split for a quick CPU smoke run (e.g. 300)",
    )

    # predict
    p_pred = sub.add_parser("predict", help="Run inference + Grad-CAM")
    p_pred.add_argument("--image", default=None, help="Path to a single image")
    p_pred.add_argument("--folder", default=None, help="Path to a folder of images")
    p_pred.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p_pred.add_argument("--save-dir", "--save_dir", dest="save_dir", default="outputs/predictions")
    p_pred.add_argument("--no-gradcam", action="store_true")

    # serve
    p_serve = sub.add_parser("serve", help="Start the REST inference API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--checkpoint", default="checkpoints/best_model.pth")

    return parser


def _cmd_train(args) -> None:
    cfg = Config.from_env()
    cfg.data.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.limit is not None:
        cfg.data.limit = args.limit
    if args.val_split is not None:
        cfg.data.val_split = args.val_split
    if args.no_resplit_val:
        cfg.data.resplit_val = False
    if args.label_smoothing is not None:
        cfg.train.label_smoothing = args.label_smoothing
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.unfreeze_epoch is not None:
        cfg.train.unfreeze_epoch = args.unfreeze_epoch
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.no_class_weights:
        cfg.train.use_class_weights = False

    from .engine import train

    train(cfg)


def _cmd_predict(args) -> None:
    from .inference import Classifier

    clf = Classifier(args.checkpoint)
    use_gradcam = not args.no_gradcam

    def _one(path: str) -> None:
        save_path = None
        if use_gradcam:
            fname = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(args.save_dir, f"{fname}_gradcam.png")
        pred = (
            clf.predict_with_gradcam(path, save_path=save_path)
            if use_gradcam
            else clf.predict(path)
        )
        logger.info("%s -> %s (%.1f%%)", os.path.basename(path), pred.label, pred.confidence * 100)

    if args.image:
        _one(args.image)
    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
        logger.info("Found %d images in '%s'", len(files), args.folder)
        for path in files:
            _one(path)
    else:
        logger.error("Provide --image <path> or --folder <path>")


def _cmd_serve(args) -> None:
    import uvicorn

    os.environ["CXR_CHECKPOINT_PATH"] = args.checkpoint
    uvicorn.run("chestxray.api:app", host=args.host, port=args.port, factory=False)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _build_parser().parse_args(argv)

    if args.command == "setup-data":
        from .dataset_setup import download_dataset

        download_dataset(args.data_dir)
    elif args.command == "eda":
        from .eda import run_eda

        run_eda(args.data_dir, args.save_dir)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "predict":
        _cmd_predict(args)
    elif args.command == "serve":
        _cmd_serve(args)


if __name__ == "__main__":
    main()
