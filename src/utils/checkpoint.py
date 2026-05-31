import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_f1,
    save_path
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "best_f1": best_f1,

        "model_state_dict":
        model.state_dict(),

        "optimizer_state_dict":
        optimizer.state_dict(),

        "scheduler_state_dict":
        scheduler.state_dict()
        if scheduler is not None
        else None
    }

    torch.save(
        checkpoint,
        save_path
    )


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if (
        scheduler is not None
        and
        checkpoint[
            "scheduler_state_dict"
        ] is not None
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    epoch = checkpoint["epoch"]

    best_f1 = checkpoint["best_f1"]

    return (
        model,
        optimizer,
        scheduler,
        epoch,
        best_f1
    )